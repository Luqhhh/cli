"""One fixed bias decode and independent CPU verification of the WFT package pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from aegis_clip.runtime import atomic_json_dump, sha256_file
from aegis_clip.submission import create_submission
from aegis_clip.v1_pipeline import load_artifact, save_artifact
from aegis_clip.v1_test_bias import fit_test_uniform_bias
from verify_v1_768_full_delivery import check_package, numpy_uniform_bias, read_predictions

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['decode', 'verify'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    cfg = json.loads(Path(args.config).read_text())
    for path, digest in cfg['sources'].items():
        require(sha256_file(path) == digest, f'Frozen source changed: {path}')
    output = Path(cfg['output'])
    preflight = json.loads((output/'preflight.json').read_text())
    inference = json.loads((output/'inference.json').read_text())
    binding = preflight['binding']
    require(binding['config_sha256'] == sha256_file(args.config) and inference['binding'] == binding
        and inference['status'] == 'inference_complete', 'Incomplete or foreign inference')
    require(inference['test_logits_sha256'] == sha256_file(output/'test_logits.pt'), 'Cache changed')
    cache = load_artifact(output/'test_logits.pt', binding)
    require(cache['checkpoint_sha256'] == sha256_file(cfg['checkpoint'])
        and cache['decoder'] == cfg['decoder'] and cache['classes'] == preflight['classes'], 'Cache lineage differs')
    matrix = cache['logits'].numpy()
    require(matrix.shape == (preflight['rows'],len(cache['classes'])) and np.isfinite(matrix).all(), 'Wrong logits')
    raw = matrix.argmax(1)
    if args.action == 'decode':
        bias = fit_test_uniform_bias(cache['logits'], iterations=200)
        save_artifact(output/'calibration.pt', dict(bias=bias, decoder=cfg['decoder'],
            test_logits_sha256=sha256_file(output/'test_logits.pt'),
            source='unlabelled_current_stage_test_logits', model_updates=0), binding)
        corrected = (cache['logits']+bias).argmax(1).numpy()
        packages = {}
        for directory, prediction, usage in [('submission_raw', raw, 'inference_only'),
                ('submission', corrected, 'unlabelled_test_marginal_bias_fitting')]:
            destination = output/directory
            create_submission([(name,cache['classes'][index]) for name,index in zip(cache['names'],prediction)],
                cache['names'], destination, cfg['checkpoint'], inference_mode='wft_fixed_sixview_sum_'+directory,
                tta_risk_acknowledged=True, valid_labels=set(cache['classes']), space_after_comma=True,
                extra_manifest=dict(binding=binding, decoder=cfg['decoder'], selected_policy=cache['selected_policy'],
                    test_usage=usage, model_updates=0, platform_score=None, status='unpromoted_candidate',
                    official_permission_confirmed=True, permission_source='user-relayed official confirmation 2026-10-01',
                    logits_sha256=sha256_file(output/'test_logits.pt'),
                    calibration_sha256=sha256_file(output/'calibration.pt') if directory=='submission' else None))
            log = check_package(ROOT,preflight['test_root'],preflight['class_mapping'],destination)
            (destination/'submission_check.log').write_text(log)
            packages[directory] = dict(csv=str(destination/'pred_results.csv'), zip=str(destination/'submission.zip'),
                csv_sha256=sha256_file(destination/'pred_results.csv'),zip_sha256=sha256_file(destination/'submission.zip'))
        atomic_json_dump(dict(status='packages_ready', packages=packages, rows=len(raw),
            raw_to_bias_changes=int(np.sum(raw!=corrected)), platform_score=None,
            seconds=time.monotonic()-started), Path(args.report))
        print('Fixed raw/bias package pair generated and checked', flush=True)
        return
    fitted = load_artifact(output/'calibration.pt', binding)
    require(fitted['decoder'] == cfg['decoder'] and fitted['source'] == 'unlabelled_current_stage_test_logits'
        and fitted['model_updates'] == 0 and fitted['test_logits_sha256'] == sha256_file(output/'test_logits.pt'), 'Bias lineage differs')
    views = cache['views'].numpy()
    summed, native = np.zeros_like(matrix), np.zeros_like(matrix)
    require(views.shape == (6,*matrix.shape) and np.isfinite(views).all(), 'Wrong per-view cache')
    for i in range(0,6,2):
        pair = views[i]+views[i+1]
        summed += pair
        native += (pair/np.float32(2))/np.float32(3)
    require(np.array_equal(summed,matrix) and np.array_equal(native,cache['native_mean'].numpy()), 'View reduction differs')
    archived = read_predictions(Path(cfg['archived_raw_csv']))
    expected_raw = [(name,cache['classes'][index]) for name,index in zip(cache['names'],raw)]
    require(archived == expected_raw and np.array_equal(native.argmax(1),raw), 'Archived raw replay differs')
    bias = fitted['bias'].numpy()
    independent = numpy_uniform_bias(matrix,200)
    error = float(np.max(np.abs(bias-independent)))
    corrected = (matrix+bias).argmax(1)
    independent_prediction = (matrix.astype(np.float64)+independent).argmax(1)
    require(error < 1e-3 and np.array_equal(corrected,independent_prediction), 'Independent FP64 bias differs')
    packages = {}
    for directory,prediction,usage in [('submission_raw',raw,'inference_only'),
            ('submission',corrected,'unlabelled_test_marginal_bias_fitting')]:
        destination = output/directory
        observed = read_predictions(destination/'pred_results.csv')
        require(observed == [(name,cache['classes'][index]) for name,index in zip(cache['names'],prediction)], 'CSV replay differs')
        with zipfile.ZipFile(destination/'submission.zip') as archive:
            require(archive.namelist() == ['pred_results.csv'] and archive.read('pred_results.csv') == (destination/'pred_results.csv').read_bytes(), 'ZIP differs')
        manifest = json.loads((destination/'manifest.json').read_text())
        require(manifest['checkpoint_sha256'] == sha256_file(cfg['checkpoint'])
            and manifest['binding'] == binding and manifest['test_usage'] == usage
            and manifest['decoder'] == cfg['decoder'] and manifest['model_updates'] == 0
            and manifest['logits_sha256'] == sha256_file(output/'test_logits.pt')
            and manifest['calibration_sha256'] == (sha256_file(output/'calibration.pt') if directory=='submission' else None), 'Package manifest differs')
        log = check_package(ROOT,preflight['test_root'],preflight['class_mapping'],destination)
        packages[directory] = dict(samples=len(observed),csv_sha256=sha256_file(destination/'pred_results.csv'),
            zip_sha256=sha256_file(destination/'submission.zip'), checker_output=log)
    require((output/'submission_raw/pred_results.csv').read_bytes() == Path(cfg['archived_raw_csv']).read_bytes(), 'Raw CSV bytes changed')
    require(sha256_file(cfg['incumbent_zip']) == cfg['incumbent_zip_sha256'], 'Incumbent changed')
    atomic_json_dump(dict(status='verified', experiment_id=cfg['experiment_id'], packages=packages,
        rows=len(raw), per_view_values_reduced=int(views.size), archived_raw_csv_byte_identical=True,
        independent_float64_bias_max_error=error, independent_prediction_matches=len(raw),
        raw_to_bias_changes=int(np.sum(raw!=corrected)), cold_rows=inference['cold_rows'],
        cold_max_abs_error=inference['cold_max_abs_error'], sources_verified=len(cfg['sources']),
        incumbent_unchanged=True, incumbent_score=cfg['incumbent_score'], platform_score=None,
        accuracy_gain_unknown=True, seconds=time.monotonic()-started), Path(args.report))
    print(f'Independent verification passed: {len(raw)} rows; bias max error {error:.8g}',flush=True)


if __name__ == '__main__':
    main()
