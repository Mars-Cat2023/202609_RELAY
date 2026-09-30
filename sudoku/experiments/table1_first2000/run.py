#!/usr/bin/env python3
"""Evaluate Table 1 on test[:2000], all 24 final EMA checkpoints, no cohort filter."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sudoku'))
sys.path.insert(0, str(ROOT / 'sudoku/xlm-core/src'))
import torch
from omegaconf import OmegaConf
from relay.model import RotaryTransformerModel, RotaryTransformerRelayModel
from relay.predictor import ConfidenceBasedPredictor
from relay.sudoku_metrics import sudoku_legal_mask
from experiments.hidden_noise_temperature.run import prepare_data, sha256
from xlm.datamodule import SimpleSpaceTokenizer

OBJECTIVES = [('MLM', 'mlm_uniform'), ('Rollout', 'rollout'),
              ('RELAY-sg', 'relay_sg'), ('RELAY', 'relay_bptt_steps2')]
PAPER = {'MLM': ['20.27 ± 0.25', '18.58 ± 2.17'],
         'Rollout': ['38.70 ± 2.09', '35.55 ± 1.15'],
         'RELAY-sg': ['58.42 ± 0.11', '59.45 ± 3.06'],
         'RELAY': ['62.67 ± 2.40', '62.07 ± 0.70']}


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load_model(path, device, tokenizer):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['hyper_parameters']
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=False)
    model_cfg = dict(cfg['model'])
    target = model_cfg.pop('_target_')
    cls = {'relay.model.RotaryTransformerModel': RotaryTransformerModel,
           'relay.model.RotaryTransformerRelayModel': RotaryTransformerRelayModel}[target]
    model_cfg.update(num_embeddings=tokenizer.full_vocab_size,
                     padding_idx=tokenizer.pad_token_id, mask_idx=tokenizer.mask_token_id)
    model = cls(**model_cfg)
    model.load_state_dict({k.removeprefix('model.'): v for k,v in ckpt['state_dict'].items()
                           if k.startswith('model.')}, strict=True)
    params = list(model.parameters())
    shadows = ckpt['ema']['shadow_params']
    assert len(params) == len(shadows)
    for p, s in zip(params, shadows):
        assert p.shape == s.shape
        p.copy_(s)
    assert ckpt['global_step'] == 300000
    pcfg = cfg['predictor']
    assert pcfg['max_steps'] == 64 and pcfg['top_k'] == 1 and pcfg['top_p'] is None
    assert pcfg['confidence'] == 'top_prob' and pcfg['threshold'] == 0.15
    with_relay = pcfg.get('with_relay', False)
    assert with_relay == (cls is RotaryTransformerRelayModel)
    model.eval().requires_grad_(False).to(device)
    predictor = ConfidenceBasedPredictor(model=model, tokenizer=tokenizer, max_steps=64,
        top_k=1, top_p=None, confidence='top_prob', threshold=0.15,
        with_relay=with_relay, log_rollout_diagnostics=False)
    return model, predictor, {'global_step': 300000, 'weights': 'ema',
        'model_target': target, 'with_relay': with_relay,
        'parameters': sum(p.numel() for p in model.parameters())}


def summarize(out, results):
    write_json(out / 'per_seed.json', results)
    if results:
        with (out / 'per_seed.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
    lines = ['# Table 1: first 2000 test puzzles, unfiltered', '',
             'Final step-300000 EMA checkpoints; exact whole-puzzle accuracy (%).',
             'Ours: mean ± sample standard deviation across training seeds (ddof=1).',
             'Paper values transcribed from the user-provided Table 1.', '',
             '| Objective | Tying | Paper (%) | Seed 1 | Seed 2 | Seed 3 | Ours mean ± SD (%) | n |',
             '|---|:---:|---:|---:|---:|---:|---:|---:|']
    groups = []
    for objective, _ in OBJECTIVES:
        for idx, tying in enumerate(['tied', 'untied']):
            group = sorted([r for r in results if r['objective'] == objective and r['tying'] == tying], key=lambda r:r['seed'])
            if not group:
                continue
            values = [r['accuracy_pct'] for r in group]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else None
            pop_sd = statistics.pstdev(values)
            seeds = {r['seed']: r['accuracy_pct'] for r in group}
            cells = ' | '.join(f'{seeds[s]:.2f}' if s in seeds else 'pending' for s in (1,2,3))
            sd_text = f'{sd:.2f}' if sd is not None else 'N/A'
            lines.append(f"| {objective} | {'✓' if tying == 'tied' else '✗'} | {PAPER[objective][idx]} | {cells} | {mean:.2f} ± {sd_text} | {len(values)} |")
            groups.append(dict(objective=objective, tying=tying, n_seeds=len(values),
                               n_puzzles_per_seed=2000, mean_accuracy_pct=mean,
                               std_sample_pct=sd, std_population_pct=pop_sd))
    write_json(out / 'table1.json', groups)
    (out / 'table1.md').write_text('\n'.join(lines) + '\n')


def reuse_rows(folder, digest, data_meta, identities, targets, tokenizer, batch_size):
    manifest = json.loads((folder / 'manifest.json').read_text())
    a = manifest['arguments']
    assert manifest['status'] == 'complete'
    assert manifest['checkpoint_sha256'] == digest
    assert manifest['data']['subset_sha256'] == data_meta['subset_sha256']
    assert a['weights'] == 'ema' and a['precision'] == 'bf16'
    assert a['batch_size'] == batch_size and a['max_steps'] == 64 and a['threshold'] == 0.15
    for name in ['sudoku/relay/model.py', 'sudoku/relay/predictor.py']:
        assert sha256(ROOT / name) == manifest['source_hashes'][name]
    assert [json.loads(x) for x in (folder / 'puzzles.jsonl').read_text().splitlines()] == identities
    grouped = {}
    for line in (folder / 'baseline.jsonl').read_text().splitlines():
        r = json.loads(line)
        assert r['temperature'] == 1 and r['sigma'] == 0
        grouped.setdefault(r['puzzle_id'], []).append(r)
    assert sorted(grouped) == list(range(2000))
    rows = []
    for i in range(2000):
        g = grouped[i]
        assert sorted(r['sample_id'] for r in g) == list(range(8))
        first = g[0]
        assert all(r['prediction_ids'] == first['prediction_ids'] for r in g)
        prediction = torch.tensor(first['prediction_ids'])
        exact = bool(torch.equal(prediction, targets[i]))
        assert all(r['exact_match'] == exact and r['clues_preserved'] for r in g)
        rows.append(dict(puzzle_id=i, prediction_ids=first['prediction_ids'], exact_match=exact,
                         legal=bool(sudoku_legal_mask(prediction.unsqueeze(0))[0]),
                         clues_preserved=True, nfe=first['nfe']))
    return rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver/test')
    parser.add_argument('--reuse', type=Path, default=ROOT / 'logs/inference_hidden_temperature/test2000_tied_seed1_ema_20260928')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260930)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Use a new output directory; prior evaluations are preserved.')
    tokenizer = SimpleSpaceTokenizer.for_numbers(vocab_size=10)
    prompts, targets, identities, data_meta = prepare_data(args.data, 0, 2000, tokenizer)
    jobs = []
    for objective, stem in OBJECTIVES:
        for tying in ('tied', 'untied'):
            for seed in (1, 2, 3):
                name = f'sudoku_extreme_{stem}_300k_{tying}_seed{seed}'
                path = ROOT / 'logs' / name / 'checkpoints/40-300000.ckpt'
                assert path.is_file(), path
                jobs.append((objective, tying, seed, name, path))
    manifest = dict(status='running', data=data_meta, checkpoint_step=300000, weights='ema',
                    n_models=len(jobs), seeds=[1,2,3], precision='bf16', batch_size=args.batch_size,
                    device=args.device, gpu=torch.cuda.get_device_name(args.device),
                    torch_version=torch.__version__, confidence='top_prob', threshold=0.15,
                    top_k=1, top_p=None, temperature=1, hidden_sigma=0, max_steps=64,
                    metric='All 81 predicted token IDs equal ground truth; one deterministic rollout per puzzle.',
                    nfe_definition='Actual forwards including original unconditional final forward and finished batch rows.',
                    source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'sudoku/relay/model.py', ROOT/'sudoku/relay/predictor.py', ROOT/'sudoku/experiments/hidden_noise_temperature/run.py']})
    write_json(out / 'manifest.json', manifest)
    (out/'puzzles.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in identities))
    results = []
    for objective, tying, seed, name, path in jobs:
        started = time.monotonic()
        digest = sha256(path)
        model, predictor, model_meta = load_model(path, args.device, tokenizer)
        assert bool(model.tie_embeddings) == (tying == 'tied') if hasattr(model, 'tie_embeddings') else True
        rows = []
        reused = objective == 'RELAY' and tying == 'tied' and seed == 1 and args.reuse.is_dir()
        if reused:
            rows = reuse_rows(args.reuse, digest, data_meta, identities, targets, tokenizer, args.batch_size)
            # Direct original-predictor check against the previously verified perturbation baseline.
            x = prompts[:args.batch_size].to(args.device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                check = predictor.predict(dict(input_ids=x, fixed=x != tokenizer.mask_token_id))
            assert check['ids'].cpu().tolist() == [r['prediction_ids'] for r in rows[:len(x)]]
            print(f'REUSED {name}: all 8 repeats audited; first batch matches original predictor.', flush=True)
        else:
            for start in range(0,2000,args.batch_size):
                end = min(start+args.batch_size,2000)
                x = prompts[start:end].to(args.device)
                fixed = x != tokenizer.mask_token_id
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    prediction = predictor.predict(dict(input_ids=x, fixed=fixed))
                ids = prediction['ids']
                clues = ((ids == x) | ~fixed).all(-1).cpu().tolist()
                assert all(clues), 'Clues changed'
                legal = sudoku_legal_mask(ids).cpu().tolist()
                cpu = ids.cpu()
                exact = (cpu == targets[start:end]).all(-1).tolist()
                for j in range(end-start):
                    rows.append(dict(puzzle_id=start+j, prediction_ids=cpu[j].tolist(),
                                     exact_match=exact[j], legal=legal[j], clues_preserved=clues[j],
                                     nfe=prediction['rollout_steps']))
                print(f'{name}: {end}/2000, elapsed {time.monotonic()-started:.1f}s', flush=True)
        assert len(rows) == 2000 and [r['puzzle_id'] for r in rows] == list(range(2000))
        assert sha256(path) == digest, 'Checkpoint changed during evaluation'
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result = dict(objective=objective, tying=tying, seed=seed, n_puzzles=2000,
                      correct=sum(r['exact_match'] for r in rows),
                      accuracy_pct=sum(r['exact_match'] for r in rows)/20,
                      legal_pct=sum(r['legal'] for r in rows)/20,
                      mean_nfe=statistics.mean(r['nfe'] for r in rows),
                      checkpoint=str(path), checkpoint_sha256=digest, **model_meta,
                      reused_from=str(args.reuse) if reused else '',
                      elapsed_seconds=time.monotonic()-started)
        results.append(result)
        summarize(out, results)
        print('RESULT '+json.dumps(result), flush=True)
        del predictor, model
        torch.cuda.empty_cache()
    manifest['status'] = 'complete'
    manifest['completed_models'] = len(results)
    write_json(out/'manifest.json', manifest)
    print((out/'table1.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
