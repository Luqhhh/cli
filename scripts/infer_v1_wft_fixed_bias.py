"""Frozen WFT FULL inference using its original, hash-bound implementation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.v1_pipeline import load_artifact, read_rows, save_artifact
from aegis_clip.v1_strategy import load_trainable_state
from v1_continuation.plan import verify
from v1_continuation.runtime import eval_loader, initial_model, require_idle_cuda


def require(value, message):
    if not value:
        raise ValueError(message)


def inspect(config_path):
    cfg = json.loads(Path(config_path).read_text())
    for name, expected in cfg['sources'].items():
        require(sha256_file(name) == expected, f'Frozen source changed: {name}')
    require(cfg['decoder'] == dict(scales=[448,512,576], crop=448, flip=True,
        reduction='sixview_logit_sum', iterations=200, strength=1.0), 'Decoder changed')
    plan, context, _ = verify(cfg['plan'])
    report = json.loads(Path(cfg['training_report']).read_text())
    require(plan['config']['route'] == 'WFT448' and plan['config']['partition'] == 'full_train', 'Wrong route')
    require(report['status'] == 'full_training_complete' and report['epochs'] == 4
        and report['plan_sha256'] == sha256_file(cfg['plan'])
        and report['artifacts'][cfg['checkpoint']] == sha256_file(cfg['checkpoint']), 'Training report differs')
    original_binding = dict(plan_sha256=sha256_file(cfg['plan']), source_binding=context.binding,
        parent_sha256=sha256_file(plan['config']['parent_checkpoint']),
        targets_sha256=sha256_file(plan['config']['targets']))
    payload = load_artifact(cfg['checkpoint'], original_binding)
    require(payload['complete'] is True and payload['epoch'] == 4 and payload['bias'] is None
        and payload['selected_policy'] == 'last_ema', 'Wrong fixed checkpoint')
    rows = read_rows(Path(context.reference['data']['dataset_manifest']).parent/'test_manifest.csv')
    names = [Path(row['image_path']).name for row in rows]
    require(len(names) == context.manifest['test_samples'] and len(set(names)) == len(names)
        and sorted(names) == sorted(p.name for p in context.test_root.iterdir() if p.is_file()), 'Test coverage differs')
    with Path(cfg['archived_raw_csv']).open() as handle:
        archived = [(name, label.strip()) for name, label in csv.reader(handle)]
    require([name for name, _ in archived] == names, 'Archived prediction order differs')
    binding = dict(config_sha256=sha256_file(config_path), original_binding=original_binding)
    return cfg, plan, context, payload, rows, archived, binding


def reductions(views):
    require(views.ndim == 3 and views.shape[0] == 6 and torch.isfinite(views).all(), 'Invalid six views')
    summed, native = torch.zeros_like(views[0]), torch.zeros_like(views[0])
    for i in range(0,6,2):
        pair = views[i] + views[i+1]
        summed += pair
        native += (pair / 2) / 3
    return summed, native


@torch.no_grad()
def collect(plan, context, model, rows, output, *, deadline):
    model.eval()
    views = torch.empty(6, len(rows), len(context.classes))
    started, last = time.monotonic(), 0.
    for scale_index, scale in enumerate([448,512,576]):
        loader = eval_loader(plan, context, rows, scale, context.test_root)
        for batch, (images, indices) in enumerate(loader, 1):
            require(time.monotonic() < deadline, 'Fixed inference time budget exhausted')
            images = images.to('cuda')
            views[2*scale_index, indices] = model(images).float().cpu()
            views[2*scale_index+1, indices] = model(images.flip(3)).float().cpu()
            now = time.monotonic()
            if now-last >= 45 or batch == len(loader):
                progress = dict(scale=scale, batch=batch, batches=len(loader), rows=len(rows), seconds=now-started)
                atomic_json_dump(progress, output/'progress.json')
                print(json.dumps(progress), flush=True)
                last = now
    require(torch.isfinite(views).all(), 'Nonfinite logits')
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'infer'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    cfg, plan, context, payload, rows, archived, binding = inspect(args.config)
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    if args.action == 'infer':
        require_idle_cuda()
        require(not (output/'test_logits.pt').exists(), 'Refusing to replace existing inference')
    model, conversion = initial_model(plan, context)
    load_trainable_state(model, payload['selected_state'])
    if args.action == 'preflight':
        atomic_json_dump(dict(status='verified', binding=binding, conversion=conversion,
            rows=len(rows), checkpoint=cfg['checkpoint'], selected_policy=payload['selected_policy'],
            test_root=str(context.test_root), class_mapping=context.reference['data']['class_mapping'],
            classes=context.classes, seconds=time.monotonic()-started), output/'preflight.json')
        print('Original plan, full checkpoint, classes and model conversion verified', flush=True)
        return
    deadline = time.monotonic() + cfg['max_inference_seconds']
    model.to('cuda').eval()
    views = collect(plan, context, model, rows, output, deadline=deadline)
    summed, native = reductions(views)
    names = [name for name, _ in archived]
    expected = torch.tensor([context.classes.index(label) for _, label in archived])
    require(torch.equal(native.argmax(1), expected), 'Native mean fails archived raw replay')
    require(torch.equal(summed.argmax(1), expected), 'SUM changes archived raw decisions')
    del model
    torch.cuda.empty_cache()
    cold, _ = initial_model(plan, context)
    load_trainable_state(cold, payload['selected_state'])
    cold.to('cuda').eval()
    replay = collect(plan, context, cold, rows[:64], output, deadline=deadline)
    cold_error = float((replay-views[:,:64]).abs().max())
    require(cold_error == 0, 'Cold checkpoint replay differs')
    save_artifact(output/'test_logits.pt', dict(views=views, logits=summed, native_mean=native,
        names=names, classes=context.classes, checkpoint_sha256=sha256_file(cfg['checkpoint']),
        decoder=cfg['decoder'], selected_policy=payload['selected_policy']), binding)
    atomic_json_dump(dict(status='inference_complete', binding=binding, decoder=cfg['decoder'],
        rows=len(rows), archived_raw_matches=len(rows), sum_raw_matches=len(rows),
        cold_rows=64, cold_max_abs_error=cold_error, conversion=conversion,
        test_logits_sha256=sha256_file(output/'test_logits.pt'), seconds=time.monotonic()-started,
        model_updates=0, platform_score=None), output/'inference.json')
    print('All raw predictions and cold per-view replay match', flush=True)


if __name__ == '__main__':
    main()
