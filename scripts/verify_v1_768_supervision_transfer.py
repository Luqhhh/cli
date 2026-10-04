"""Independent recount, frozen-visual lineage and full CSV/ZIP verification."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import torch

from verify_v1_task_neighbors import require, sha, read, rows, unit, counts, compare, equal

ROOT=Path(__file__).resolve().parents[1]
ARMS=('control512','candidate768')


def verify(config, validation_only=False):
    cfg=read(config); out=Path(cfg['output']); pre=read(out/'preflight.json')
    fit=read(out/'fit_report.json')
    delivery=None if validation_only else read(out/'delivery_report.json')
    require(pre['binding']==fit['binding'],'Bindings differ')
    if delivery is not None:
        require(delivery['status']=='completed_verified_delivery','Delivery incomplete')
        require(pre['binding']==delivery['binding'],'Delivery binding differs')
    for path,digest in pre['sources'].items(): require(sha(path)==digest,f'Source changed: {path}')
    require(sha(out/'frozen_validation.npz')==pre['frozen_validation_sha256'],'Frozen groups changed')
    stage=Path('/home/lux1/noise/artifacts/stages/repechage/20260921')
    train,val,full=[rows(stage/f'{n}.csv') for n in ('train_dev','val_dev','full_train')]
    require(not {r['content_group'] for r in train}&{r['content_group'] for r in val},'Content leakage')
    labels=np.array([int(r['label']) for r in val]); paths=np.array([r['image_path'] for r in val])
    mapping=read(stage/'class_to_idx.json'); classes=[k for k,v in sorted(mapping.items(),key=lambda p:p[1])]
    z=np.load(out/'frozen_validation.npz',allow_pickle=False); pred=np.load(out/'validation_predictions.npz',allow_pickle=False)
    for payload in (z,pred):
        require(np.array_equal(payload['labels'],labels) and np.array_equal(payload['image_paths'],paths),'Validation identity differs')
    new_info=read(out/'target_comparison.json')
    require(sha(new_info['targets'])==new_info['targets_sha256']==fit['new_targets_sha256'],'New target digest differs')
    import yaml
    original=yaml.safe_load((ROOT/cfg['crt_config']).read_text())
    old=torch.load(original['targets'],map_location='cpu',weights_only=False)
    new=torch.load(new_info['targets'],map_location='cpu',weights_only=False)
    for t in (old,new):
        require(t['image_paths']==[r['image_path'] for r in train] and t['class_names']==classes,'Target population differs')
        require(np.array_equal(t['labels'].numpy(),[int(r['label']) for r in train]),'Target source labels differ')
    a,b=old['weights'].numpy()>0,new['weights'].numpy()>0; shared=a&b
    changes=dict(old_active=int(sum(a)),new_active=int(sum(b)),union_active=int(sum(a|b)),added=int(sum(~a&b)),
        removed=int(sum(a&~b)),shared=int(sum(shared)),
        changed_targets_shared=int(sum(shared&(old['targets'].numpy()!=new['targets'].numpy()))),
        changed_weights=int(sum(old['weights'].numpy()!=new['weights'].numpy())),
        absolute_weight_delta_sum=float(np.abs(old['weights'].numpy().astype('float64')-new['weights'].numpy()).sum()),
        old_pseudo=int(sum(old['pseudo'].numpy())),new_pseudo=int(sum(new['pseudo'].numpy())),
        old_mass=np.bincount(old['targets'].numpy(),weights=old['weights'].numpy(),minlength=len(classes)).tolist(),
        new_mass=np.bincount(new['targets'].numpy(),weights=new['weights'].numpy(),minlength=len(classes)).tolist())
    for key,value in changes.items():
        if isinstance(value,float): require(abs(value-new_info['changes'][key])<1e-9,f'Target recount differs: {key}')
        else: require(value==new_info['changes'][key],f'Target recount differs: {key}')
    order=torch.load(out/'shared_epoch_orders.pt',map_location='cpu',weights_only=True).numpy()
    population=np.where(a|b)[0]
    require(order.shape==(cfg['head_fit']['epochs'],len(population)),'Wrong training schedule')
    for epoch in order: require(np.array_equal(np.sort(epoch),population),'A paired epoch lost or duplicated rows')
    require(fit['evidence'][ARMS[0]]['dropout_rng_sha256']==fit['evidence'][ARMS[1]]['dropout_rng_sha256'],'Dropout pairing differs')
    import hashlib
    for arm in ARMS:
        require(fit['evidence'][arm]['sample_order_sha256']==hashlib.sha256(order.tobytes()).hexdigest(),'Schedule digest differs')
        require(fit['evidence'][arm]['steps']==len(order)*int(np.ceil(len(population)/cfg['head_fit']['batch_size'])),'Step count differs')
    cache=torch.load(Path(original['output'])/'frozen_features768.pt',map_location='cpu',weights_only=False)
    by_path={p:i for i,p in enumerate(cache['image_paths'])}; x=unit(cache['features'][[by_path[p] for p in paths]].numpy())
    parent=torch.load(original['parent_checkpoint'],map_location='cpu',weights_only=False)
    for arm in ARMS:
        head=torch.load(out/f'head_{arm}.pt',map_location='cpu',weights_only=True)
        replay=(x@unit(head['weight'].numpy()).T).argmax(1)
        require(np.array_equal(replay,pred[arm]),f'Independent FP64 validation head replay differs: {arm}')
        selected=torch.load(out/arm/'selected.pt',map_location='cpu',weights_only=False)
        require(selected['model_config']==dict(parent['model_config'],feature_path='pre_projection'),'Candidate architecture differs')
        for key,value in parent['selected_state'].items():
            if not key.startswith('head.'):
                require(torch.equal(value,selected['selected_state'][key]),'Visual tensor changed')
        for key,value in head.items(): require(torch.equal(value,selected['selected_state']['head.'+key]),'Exported head differs')
    grouping=defaultdict(set)
    for r in val: grouping[r['content_group']].add(r['label'])
    train_counts=np.bincount([int(r['label']) for r in train],minlength=len(classes))
    tail=np.lexsort((np.arange(len(classes)),train_counts))[:len(classes)//10]
    require(np.array_equal(z['tail75'],np.isin(labels,tail)),'Tail group differs')
    require(np.array_equal(z['content_conflict'],[len(grouping[r['content_group']])>1 for r in val]),'Content-conflict group differs')
    for group in pre['group_sizes']:
        mask=z[group]; require(int(mask.sum())==pre['group_sizes'][group],'Group size differs')
        for arm,p in dict(archived_crt=z['archived_crt'],**{k:pred[k] for k in ARMS}).items():
            equal(fit['metrics'][group][arm],counts(labels,p,mask),f'metric/{group}/{arm}')
        for key,before,after in [('candidate_vs_control',pred[ARMS[0]],pred[ARMS[1]]),
                                ('candidate_vs_archived',z['archived_crt'],pred[ARMS[1]]),
                                ('control_vs_archived',z['archived_crt'],pred[ARMS[0]])]:
            equal(fit['comparisons'][group][key],compare(labels,before,after,mask),f'pair/{group}/{key}')
    gate=cfg['gate']; pairs=fit['comparisons']; m=fit['metrics']['all']
    gates=dict(net_vs_control=pairs['all']['candidate_vs_control']['net']>=gate['net_vs_control_min'],
        net_vs_archived=pairs['all']['candidate_vs_archived']['net']>=gate['net_vs_archived_crt_min'],
        common_net=pairs['common_candidate_wrong']['candidate_vs_control']['net']>=gate['common_net_vs_control_min'],
        macro_delta=m[ARMS[1]]['macro']-m[ARMS[0]]['macro']>=gate['macro_delta_vs_control_min'])
    require(gates==fit['gates'],'Gate decision differs')
    require(fit['decision']==('supports_review_not_automatic_full' if all(gates.values()) else 'close_fixed_supervision_transfer'),'Decision differs')
    if validation_only:
        return dict(status='validation_verified_delivery_pending',experiment_id=cfg['experiment_id'],
            source_files=len(pre['sources']),all_train_targets_recounted=len(train),paired_orders_checked=list(order.shape),
            fp64_val_predictions_replayed=2*len(val),metrics_recounted=12,comparisons_recounted=12,gates=gates,
            decision=fit['decision'],new_candidate_delivery_complete=False,platform_score=None)
    require(delivery['decision']==fit['decision'],'Delivery decision differs')
    packages=[]
    for arm in ARMS:
        cached=torch.load(out/arm/'test_logits.pt',map_location='cpu',weights_only=False)
        calibration=torch.load(out/arm/'test_calibration.pt',map_location='cpu',weights_only=False)
        require(cached['checkpoint_sha256']==calibration['checkpoint_sha256']==sha(out/arm/'selected.pt'),'Test model lineage differs')
        require(calibration['test_logits_sha256']==sha(out/arm/'test_logits.pt'),'Bias cache lineage differs')
        require(cached['decoder']==calibration['decoder']==dict(scales=[448,512,576],flip=True,bias_source='test_uniform_experimental',experimental_test_bias=True,logit_reduction='sum',bias_iterations=200,bias_strength=1.0),'Decoder differs')
        require(cached['names']==[Path(r['image_path']).name for r in rows(stage/'test_manifest.csv')],'Test names differ')
        for directory,bias in [('submission_raw',0),('submission',calibration['bias'].numpy())]:
            p=(cached['logits'].numpy()+bias).argmax(1)
            expected=''.join(f'{name}, {classes[int(i)]}\r\n' for name,i in zip(cached['names'],p)).encode()
            package=out/arm/directory
            require((package/'pred_results.csv').read_bytes()==expected,'CSV replay differs')
            with zipfile.ZipFile(package/'submission.zip') as archive:
                require(archive.namelist()==['pred_results.csv'] and archive.read('pred_results.csv')==expected,'ZIP replay differs')
            run=subprocess.run([sys.executable,str(ROOT/'scripts/check_submission.py'),'--test_dir','/home/lux1/noise/test',
                '--class-mapping',str(stage/'class_to_idx.json'),'--csv',str(package/'pred_results.csv'),'--zip',str(package/'submission.zip')],capture_output=True,text=True,check=True)
            require('All checks passed' in run.stdout+run.stderr,'Formal checker failed')
            packages.append(dict(arm=arm,decoder=directory,rows=len(p),zip=str(package/'submission.zip'),zip_sha256=sha(package/'submission.zip'),nine_checks_passed=True))
    baseline=read(ROOT/'results/top4_platform_20261004/record.json')['reported_packages'][0]
    require(sha(baseline['desktop_path'])==baseline['zip_sha256'],'Incumbent changed')
    require(read(out/'cost.json')['within_budget'] and fit['seconds']+delivery['seconds']<=cfg['budget_seconds']['fit_and_delivery'],'Fixed cost budget exceeded')
    return dict(status='independently_verified',experiment_id=cfg['experiment_id'],source_files=len(pre['sources']),
        all_train_targets_recounted=len(train),paired_orders_checked=list(order.shape),fp64_val_predictions_replayed=2*len(val),
        metrics_recounted=12,comparisons_recounted=12,gates=gates,decision=fit['decision'],packages=packages,
        incumbent_unchanged=True,platform_score=None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--output',type=Path)
    parser.add_argument('--validation-only',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(2); result=verify(args.config,args.validation_only)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
