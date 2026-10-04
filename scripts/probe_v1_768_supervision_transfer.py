"""Fixed train_dev supervision transfer to a frozen task-trained 768 visual tower."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import (StageContext, load_recipe, load_artifact, save_artifact,
    prepare_targets, seed_training, image_transform, loader_for, read_rows, fingerprint)
from aegis_clip.v1_strategy import CosineHead, build_classifier, load_trainable_state, target_probabilities
from aegis_clip.v1_test_bias import infer_test_uniform
from run_v1_post768_job import read_config, inspect, HEAD_FIT, DECODER, IndependentHeads, context_for, verify_package
from diagnose_v1_candidate_errors import metrics, paired

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('control512', 'candidate768')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def targets_valid(targets, rows, classes):
    require(targets['image_paths'] == [r['image_path'] for r in rows], 'Target row identity differs')
    require(targets['class_names'] == classes, 'Target classes differ')
    labels = torch.tensor([int(r['label']) for r in rows])
    require(torch.equal(targets['labels'], labels), 'Target original labels differ')
    n, c = len(rows), len(classes)
    for key in ('targets', 'weights', 'original_alpha', 'kept', 'pseudo', 'agreement'):
        require(targets[key].shape == (n,), f'Target shape differs: {key}')
    require(torch.isfinite(targets['weights']).all() and (targets['weights'] >= 0).all(), 'Invalid weights')
    require(((targets['targets'] >= 0) & (targets['targets'] < c)).all(), 'Invalid target labels')
    mass = torch.zeros(c).scatter_add_(0, targets['targets'], targets['weights'])
    require((mass > 0).all(), 'A class lost all reliable supervision')


def target_changes(old, new, classes):
    a, b = old['weights'] > 0, new['weights'] > 0
    return dict(old_active=int(a.sum()), new_active=int(b.sum()), union_active=int((a|b).sum()),
        added=int((~a&b).sum()), removed=int((a&~b).sum()), shared=int((a&b).sum()),
        changed_targets_shared=int((a&b&(old['targets'] != new['targets'])).sum()),
        changed_weights=int((old['weights'] != new['weights']).sum()),
        absolute_weight_delta_sum=float((old['weights'].double()-new['weights'].double()).abs().sum()),
        old_pseudo=int(old['pseudo'].sum()), new_pseudo=int(new['pseudo'].sum()),
        old_mass=torch.zeros(classes, dtype=torch.float64).scatter_add_(0, old['targets'], old['weights'].double()).tolist(),
        new_mass=torch.zeros(classes, dtype=torch.float64).scatter_add_(0, new['targets'], new['weights'].double()).tolist())


def common_population(old, new):
    require(len(old['weights']) == len(new['weights']), 'Population lengths differ')
    active = torch.where((old['weights'] > 0) | (new['weights'] > 0))[0]
    require(len(active) > 0, 'Empty paired population')
    return active


def weighted_loss(logits, targets, indices, classes, smoothing):
    weights = targets['weights'][indices]
    require(bool(weights.sum() > 0), 'Paired batch has zero supervision mass')
    prob = target_probabilities(targets['targets'][indices], targets['labels'][indices],
                                targets['original_alpha'][indices], classes, smoothing)
    return (-(prob*logits.log_softmax(1)).sum(1)*weights).sum()/weights.sum()


def load_inputs(config):
    cfg = read(config)
    require(cfg['head_fit'] == HEAD_FIT and cfg['sampler'] == 'union_positive_old_new_fixed_shared_epoch_permutation',
            'Only the fixed paired head protocol is supported')
    require(cfg['automatic_full_training'] is False and cfg['platform_upload'] is False, 'Scope differs')
    sources = {}
    def checked(path, expected=None):
        path = Path(path).resolve(); digest = sha256_file(path)
        require(expected is None or digest == expected, f'Input digest differs: {path}')
        sources[str(path)] = digest
        return path
    checked(config); checked(__file__)
    old_cfg = read_config(checked(ROOT/cfg['crt_config']))
    ctx, parent, old, _ = inspect(old_cfg)
    target_ctx = StageContext(load_recipe(checked(ROOT/cfg['target_recipe'])))
    require(target_ctx.config['source']['partition'] == 'train_dev' and target_ctx.train == ctx.train
        and target_ctx.val == ctx.val and target_ctx.classes == ctx.classes, 'New supervision must use isolated train_dev')
    targets_valid(old, ctx.train, ctx.classes)
    src = Path(old_cfg['output']); report = read(checked(ROOT/cfg['crt_report']))
    require(report == read(checked(src/'report.json')) and report['validation_independent']
        and report['visual_parameter_updates'] is False, 'Archived CRT report differs')
    cache_path = checked(src/'frozen_features768.pt', cfg['feature_sha256'])
    cache = load_artifact(cache_path, report['binding'])
    checked(cache_path.with_suffix('.sha256.json'))
    require(cache['image_paths'] == [r['image_path'] for r in ctx.full] and cache['features'].shape == (len(ctx.full),768)
        and cache['image_size'] == 448 and cache['visual_parameter_updates'] is False, 'Task feature identity differs')
    require(torch.isfinite(cache['features']).all(), 'Nonfinite task features')
    for path, digest in [(old_cfg['targets'], report['binding']['targets_sha256']),
                         (old_cfg['parent_checkpoint'], report['binding']['parent_checkpoint_sha256'])]:
        checked(path, digest); checked(Path(path).with_suffix('.sha256.json'))
    checked(old_cfg['parent_config']); checked(ctx.official, ctx.binding['official_checkpoint_sha256'])
    source_cache = target_ctx.config['source']['preprojection_cache']
    checked(source_cache['tensor_path'], source_cache['tensor_sha256'])
    checked(source_cache['protocol_path'], source_cache['protocol_sha256'])
    base = Path(ctx.reference['data']['dataset_manifest']).parent
    for name in ('dataset_manifest.json','class_to_idx.json','train_dev.csv','val_dev.csv','full_train.csv'):
        checked(base/name)
    index = {p:i for i,p in enumerate(cache['image_paths'])}
    train_i = torch.tensor([index[r['image_path']] for r in ctx.train])
    val_i = torch.tensor([index[r['image_path']] for r in ctx.val])
    closure = read(checked(ROOT/cfg['crt_closure']))
    z = np.load(checked(src/'validation_predictions.npz', closure['validation_npz_sha256']))
    labels = np.array([int(r['label']) for r in ctx.val]); paths = np.array([r['image_path'] for r in ctx.val])
    require(np.array_equal(labels,z['labels']) and np.array_equal(paths,z['image_paths']), 'Archived validation differs')
    selected = load_artifact(checked(src/'unbalanced/selected.pt', report['packages']['unbalanced']['checkpoint_sha256']),
                            read(src/'unbalanced/selected.sha256.json')['binding'])
    head = CosineHead(768,len(ctx.classes)); head.load_state_dict({k[5:]:v for k,v in selected['selected_state'].items() if k.startswith('head.')})
    with torch.no_grad():
        native = torch.cat([head(x) for x in cache['features'][val_i].split(1024)]).argmax(1).numpy()
    require(np.array_equal(native,z['unbalanced768']), 'All archived CRT predictions must replay')
    reflection = read(checked(ROOT/cfg['common_source']))
    common_path = next(p for p in reflection['artifacts'] if Path(p).name == 'lr512_swa.npz')
    common = np.load(checked(common_path, reflection['artifacts'][common_path]))
    require(np.array_equal(common['labels'],labels) and np.array_equal(common['image_paths'],paths), 'Common error rows differ')
    by_group = {}
    for r in ctx.val: by_group.setdefault(r['content_group'],set()).add(r['label'])
    groups = dict(all=np.ones(len(labels),bool), common_candidate_wrong=common['common_candidate_wrong'].copy(),
        tail75=np.isin(labels,z['tail']), content_conflict=np.array([len(by_group[r['content_group']])>1 for r in ctx.val]))
    require(groups['common_candidate_wrong'].sum()==2875 and (native[groups['common_candidate_wrong']] != labels[groups['common_candidate_wrong']]).all(),
            'Frozen common-error definition differs')
    for module in ('v1_pipeline.py','v1_strategy.py','v1_test_bias.py','prior_alignment.py'):
        checked(ROOT/'reproducibility/aegis_f1/aegis_clip'/module)
    binding = dict(ctx.binding, recipe_sha256=fingerprint(cfg), transfer_sources_sha256=fingerprint(sources))
    return dict(cfg=cfg,ctx=ctx,target_ctx=target_ctx,parent=parent,old=old,cache=cache,src=src,old_cfg=old_cfg,
                train_i=train_i,val_i=val_i,labels=labels,paths=paths,native=native,groups=groups,sources=sources,binding=binding)


def prepare(config):
    d=load_inputs(config); cfg=d['cfg']; out=Path(cfg['output']); out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'frozen_validation.npz',labels=d['labels'],image_paths=d['paths'],archived_crt=d['native'],**d['groups'])
    atomic_json_dump(dict(experiment_id=cfg['experiment_id'],status='prepared',binding=d['binding'],sources=d['sources'],
        train_rows=len(d['ctx'].train),val_rows=len(d['ctx'].val),parent_partition='train_dev',
        frozen_validation_sha256=sha256_file(out/'frozen_validation.npz'),archived_predictions_replayed=len(d['labels']),
        group_sizes={g:int(v.sum()) for g,v in d['groups'].items()},gate=cfg['gate'],budget_seconds=cfg['budget_seconds']),out/'preflight.json')
    print(json.dumps(dict(status='prepared',output=str(out),train_rows=len(d['ctx'].train),val_rows=len(d['ctx'].val))),flush=True)


def checked_preflight(config):
    d=load_inputs(config); out=Path(d['cfg']['output']); pre=read(out/'preflight.json')
    require(pre['sources']==d['sources'] and pre['binding']==d['binding'], 'Preparation source binding changed')
    require(sha256_file(out/'frozen_validation.npz')==pre['frozen_validation_sha256'], 'Frozen evaluation changed')
    return d,out


def targets(config):
    d,out=checked_preflight(config); started=time.monotonic()
    require(torch.cuda.is_available(),'Local CUDA unavailable')
    path=prepare_targets(d['target_ctx'],'cuda')
    new=load_artifact(path,d['target_ctx'].binding); targets_valid(new,d['ctx'].train,d['ctx'].classes)
    change=target_changes(d['old'],new,len(d['ctx'].classes)); seconds=time.monotonic()-started
    require(change['changed_weights']>0 or change['changed_targets_shared']>0,'No supervision intervention')
    result=dict(status='targets_ready',seconds=seconds,targets=str(path),targets_sha256=sha256_file(path),
        changes=change,within_budget=seconds<=d['cfg']['budget_seconds']['targets'],
        train_rows=len(d['ctx'].train),val_rows_used_in_fitting=0,test_rows_used=0,platform_gain_known=False)
    atomic_json_dump(result,out/'target_comparison.json'); print(json.dumps({k:v for k,v in result.items() if k!='changes'}),flush=True)
    require(result['within_budget'],'Target preparation exceeded fixed cost budget')


def fit_heads(x, target_pair, initial, active, fit, *, device, directory=None, cost_batches=None):
    x=x.to(device); heads={}; evidence={}; val_order=None; rng_proof=None
    for arm,t in zip(ARMS,target_pair):
        seed_training(fit['seed']); head=CosineHead(x.shape[1],initial['weight'].shape[0],dropout=fit['dropout']).to(device)
        head.load_state_dict(initial); opt=torch.optim.AdamW(head.parameters(),lr=fit['lr'],weight_decay=fit['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,fit['epochs'],eta_min=fit['minimum_lr'])
        generator=torch.Generator(device=device).manual_seed(fit['seed']); pool=active.to(device)
        t={k:t[k].to(device) for k in ('weights','targets','labels','original_alpha')}
        orders=[]; dropout=hashlib.sha256(); history=[]; steps=0
        for epoch in range(1,(1 if cost_batches else fit['epochs'])+1):
            order=pool[torch.randperm(len(pool),device=device,generator=generator)]; orders.append(order.cpu())
            losses=[]; head.train()
            for indices in order.split(fit['batch_size']):
                opt.zero_grad(set_to_none=True)
                state=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else torch.get_rng_state()
                dropout.update(state.cpu().numpy().tobytes())
                loss=weighted_loss(head(x[indices]),t,indices,len(initial['weight']),fit['label_smoothing'])
                require(bool(torch.isfinite(loss)),'Nonfinite paired head loss')
                loss.backward(); opt.step(); losses.append(float(loss.detach())); steps+=1
                if cost_batches and steps>=cost_batches: break
            scheduler.step(); history.append(dict(epoch=epoch,steps=steps,mean_loss=float(np.mean(losses))))
        schedule=torch.stack(orders)
        if val_order is None: val_order=schedule; rng_proof=dropout.hexdigest()
        require(torch.equal(val_order,schedule) and rng_proof==dropout.hexdigest(),'Paired sample order or dropout RNG differs')
        heads[arm]=head.cpu().eval(); evidence[arm]=dict(steps=steps,sample_order_sha256=tensor_sha(schedule),
            dropout_rng_sha256=dropout.hexdigest(),history=history,initial_sha256={k:tensor_sha(v) for k,v in initial.items()})
        if directory: torch.save(heads[arm].state_dict(),directory/f'head_{arm}.pt')
    if directory: torch.save(val_order,directory/'shared_epoch_orders.pt')
    return heads,evidence


def fit(config):
    d,out=checked_preflight(config); cfg=d['cfg']; start=time.monotonic()
    require(not (out/'fit_report.json').exists(),'Fit already exists')
    info=read(out/'target_comparison.json'); require(info['within_budget'],'Target cost gate failed')
    require(sha256_file(info['targets'])==info['targets_sha256'],'New targets changed')
    new=load_artifact(info['targets'],d['target_ctx'].binding); targets_valid(new,d['ctx'].train,d['ctx'].classes)
    active=common_population(d['old'],new); x=d['cache']['features'][d['train_i']]
    cost_start=time.monotonic()
    _,cost_evidence=fit_heads(x,(d['old'],new),d['cache']['initial_head'],active,cfg['head_fit'],device='cuda',cost_batches=4)
    cost_seconds=time.monotonic()-cost_start; torch.cuda.empty_cache()
    estimate=cost_seconds/8*2*cfg['head_fit']['epochs']*((len(active)+8191)//8192)
    # Archived CRT total includes feature extraction, so it is a conservative delivery allowance.
    delivery_allowance=float(read(ROOT/cfg['crt_report'])['elapsed_seconds'])
    atomic_json_dump(dict(status='cost_complete_discarded_heads',seconds=cost_seconds,estimated_fit_seconds=estimate,
        delivery_allowance_seconds=delivery_allowance,within_budget=estimate+delivery_allowance<cfg['budget_seconds']['fit_and_delivery'],
        evidence=cost_evidence),out/'cost.json')
    require(estimate+delivery_allowance<cfg['budget_seconds']['fit_and_delivery'],'Fixed cost gate failed')
    heads,evidence=fit_heads(x,(d['old'],new),d['cache']['initial_head'],active,cfg['head_fit'],device='cuda',directory=out)
    predictions={}; classes=len(d['ctx'].classes)
    with torch.no_grad():
        for arm,head in heads.items():
            predictions[arm]=torch.cat([head(chunk) for chunk in d['cache']['features'][d['val_i']].split(1024)]).argmax(1).numpy()
    comparisons={}; all_metrics={}
    for group,mask in d['groups'].items():
        all_metrics[group]={arm:metrics(d['labels'],pred,classes,mask) for arm,pred in dict(archived_crt=d['native'],**predictions).items()}
        comparisons[group]=dict(candidate_vs_control=paired(d['labels'],predictions[ARMS[0]],predictions[ARMS[1]],mask),
            candidate_vs_archived=paired(d['labels'],d['native'],predictions[ARMS[1]],mask),
            control_vs_archived=paired(d['labels'],d['native'],predictions[ARMS[0]],mask))
    gate=cfg['gate']; all_pair=comparisons['all']
    gates=dict(net_vs_control=all_pair['candidate_vs_control']['net']>=gate['net_vs_control_min'],
        net_vs_archived=all_pair['candidate_vs_archived']['net']>=gate['net_vs_archived_crt_min'],
        common_net=comparisons['common_candidate_wrong']['candidate_vs_control']['net']>=gate['common_net_vs_control_min'],
        macro_delta=all_metrics['all'][ARMS[1]]['macro']-all_metrics['all'][ARMS[0]]['macro']>=gate['macro_delta_vs_control_min'])
    np.savez_compressed(out/'validation_predictions.npz',labels=d['labels'],image_paths=d['paths'],**predictions)
    result=dict(status='fitted_pending_delivery',binding=d['binding'],evidence=evidence,metrics=all_metrics,comparisons=comparisons,
        gates=gates,decision='supports_review_not_automatic_full' if all(gates.values()) else 'close_fixed_supervision_transfer',
        seconds=time.monotonic()-start,new_targets_sha256=info['targets_sha256'],visual_parameter_updates=0,
        validation_independent=True,platform_score=None)
    atomic_json_dump(result,out/'fit_report.json'); print(json.dumps(dict(status=result['status'],metrics=all_metrics['all'],gates=gates)),flush=True)


@torch.no_grad()
def deliver(config):
    d,out=checked_preflight(config); cfg=d['cfg']; ctx=d['ctx']; parent=d['parent']; start=time.monotonic()
    fit_report=read(out/'fit_report.json'); require(fit_report['binding']==d['binding'],'Fit lineage differs')
    require(not (out/'delivery_report.json').exists(),'Delivery already exists')
    heads={arm:CosineHead(768,len(ctx.classes),dropout=cfg['head_fit']['dropout']) for arm in ARMS}
    model_config=dict(parent['model_config'],feature_path='pre_projection'); contexts={}; checkpoints={}
    body={k:v for k,v in parent['selected_state'].items() if not k.startswith('head.')}
    for arm,head in heads.items():
        head.load_state_dict(torch.load(out/f'head_{arm}.pt',map_location='cpu',weights_only=True)); head.eval()
        dest=out/arm; dest.mkdir(exist_ok=False)
        arm_binding=dict(d['binding'],arm=arm,targets_sha256=sha256_file(cfg_target(d,arm)),
                         fit_report_sha256=sha256_file(out/'fit_report.json'))
        current=context_for(ctx,cfg,arm_binding,dest,model_config)
        checkpoint=dest/'selected.pt'; state=dict(body,**{'head.'+k:v for k,v in head.state_dict().items()})
        save_artifact(checkpoint,dict(selected_state=state,classes=ctx.classes,model_config=model_config,
            selected_policy='frozen_visual_supervision_transfer_last20',parent_checkpoint=d['old_cfg']['parent_checkpoint'],
            targets_sha256=arm_binding['targets_sha256'],head_epochs=20,visual_parameter_updates=False),arm_binding)
        contexts[arm],checkpoints[arm]=current,checkpoint
    model=build_classifier(ctx.official,len(ctx.classes),model_config,'cuda').eval()
    load_trainable_state(model,load_artifact(checkpoints[ARMS[0]],contexts[ARMS[0]].binding)['selected_state'])
    predictor=IndependentHeads(model.visual,[heads[a].to('cuda') for a in ARMS]).eval()
    examples=torch.cat([images for images,_ in loader_for(ctx,ctx.val[:64],image_transform(448))]).to('cuda')
    cold={}
    for arm_no,arm in enumerate(ARMS):
        standalone=build_classifier(ctx.official,len(ctx.classes),model_config,'cuda').eval()
        load_trainable_state(standalone,load_artifact(checkpoints[arm],contexts[arm].binding)['selected_state'])
        error=0.; predictions=[]
        for chunk in examples.split(ctx.config['train']['batch_size']):
            actual=standalone(chunk); shared=predictor(chunk)[:,arm_no*len(ctx.classes):(arm_no+1)*len(ctx.classes)]
            error=max(error,float((actual-shared).abs().max())); predictions.extend(actual.argmax(1).tolist())
        require(error<=2e-5,'Shared test forward differs from standalone checkpoint')
        expected=np.load(out/'validation_predictions.npz')[arm][:64]
        require(np.array_equal(predictions,expected),'Cold validation predictions differ from cached features')
        cold[arm]=dict(images=64,max_error=error,predictions_equal=True); del standalone
    base=Path(ctx.reference['data']['dataset_manifest']).parent; rows=read_rows(base/'test_manifest.csv')
    names=[Path(r['image_path']).name for r in rows]; nclasses=len(ctx.classes)
    logits=torch.zeros((len(rows),2*nclasses)); last=time.monotonic()
    for scale in DECODER['scales']:
        loader=loader_for(ctx,rows,image_transform(448,scale=scale),root=ctx.test_root)
        for batch,(images,indices) in enumerate(loader):
            images=images.to('cuda'); value=predictor(images)+predictor(images.flip(3)); logits[indices]+=value.cpu()
            if time.monotonic()-last>30 or batch==len(loader)-1:
                progress=dict(stage='delivery_inference',scale=scale,batch=batch+1,batches=len(loader),seconds=time.monotonic()-start)
                atomic_json_dump(progress,out/'progress.json'); print(json.dumps(progress),flush=True); last=time.monotonic()
            require(time.monotonic()-start+fit_report['seconds']<=cfg['budget_seconds']['fit_and_delivery'],'Fixed delivery time budget exceeded')
    del predictor,model; packages={}
    for i,arm in enumerate(ARMS):
        current=contexts[arm]; dest=out/arm
        save_artifact(dest/'test_logits.pt',dict(logits=logits[:,i*nclasses:(i+1)*nclasses].contiguous(),names=names,
            class_names=ctx.classes,decoder=DECODER,checkpoint_sha256=sha256_file(checkpoints[arm]),
            selected_policy='frozen_visual_supervision_transfer_last20'),current.binding)
        infer_test_uniform(current,checkpoints[arm],'cuda'); packages[arm]=verify_package(current,checkpoints[arm],dest)
        require(packages[arm]['independent_refit_prediction_matches']==len(rows),'Independent bias changes hard decisions')
    result=dict(status='completed_verified_delivery',experiment_id=cfg['experiment_id'],binding=d['binding'],
        packages=packages,cold_replay=cold,seconds=time.monotonic()-start,decision=fit_report['decision'],
        platform_score=None,incumbent_unchanged=True)
    atomic_json_dump(result,out/'delivery_report.json'); print(json.dumps(result),flush=True)


def cfg_target(data,arm):
    return data['old_cfg']['targets'] if arm==ARMS[0] else Path(data['target_ctx'].config['output']['root'])/'targets.pt'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','targets','fit','deliver'])
    parser.add_argument('--config',required=True)
    args=parser.parse_args(); torch.set_num_threads(2)
    globals()[args.action](args.config)
