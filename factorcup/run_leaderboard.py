#!/usr/bin/env python3
"""
Meta-runner: scores every standalone factoring function in the repo
in PARALLEL (4 workers) and writes results incrementally to LEADERBOARD.md.

Each entry runs as its own python subprocess of `_score_one.py`, which spawns
its own isolated subprocesses for the actual factor() call. So we use 4 cores
at once for orchestration; the inner subprocesses also use 1 core each.

Run from factorcup/:
    /root/polynomial-time/venv/bin/python run_leaderboard.py
"""

import os
import sys
import time
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, 'venv', 'bin', 'python')

PARALLELISM = 4

# (display_name, location, source_module, function_name)
ENTRIES = [
    ('entry', 'native', None, None),

    ('trial_division',     'fc', 'baselines',  'trial_division'),
    ('pollard_rho',        'fc', 'baselines',  'pollard_rho'),
    ('quadratic_sieve',    'fc', 'baselines',  'quadratic_sieve'),

    ('multibase_power_gcd',  'root', 'creative', 'multibase_power_gcd'),
    ('coppersmith_bootstrap','root', 'creative', 'coppersmith_bootstrap'),
    ('ecm_attack',           'root', 'creative', 'ecm_attack'),
    ('quadratic_form_attack','root', 'creative', 'quadratic_form_attack'),
    ('multi_polynomial_sieve','root','creative', 'multi_polynomial_sieve'),
    ('structured_walk_attack','root','creative', 'structured_walk_attack'),
    ('creative_combined',    'root', 'creative', 'creative_combined'),

    ('lattice_partial_info',     'root', 'experimental', 'lattice_partial_info'),
    ('classical_period_attack',  'root', 'experimental', 'classical_period_attack'),
    ('character_sum_attack',     'root', 'experimental', 'character_sum_attack'),
    ('cfrac_attack',             'root', 'experimental', 'cfrac_attack'),
    ('spectral_attack',          'root', 'experimental', 'spectral_attack'),
    ('combined_attack',          'root', 'experimental', 'combined_attack'),
    ('modular_constraint_attack','root', 'experimental', 'modular_constraint_attack'),
    ('group_structure_attack',   'root', 'experimental', 'group_structure_attack'),

    ('k01_power_residue',     'kill', 'k01_novel', 'power_residue_attack'),
    ('k01_coppersmith',       'kill', 'k01_novel', 'coppersmith_attack'),
    ('k01_character_matrix',  'kill', 'k01_novel', 'character_matrix_attack'),
    ('k01_algebraic_norm',    'kill', 'k01_novel', 'algebraic_norm_attack'),
    ('k01_trace_sequence',    'kill', 'k01_novel', 'trace_sequence_attack'),
    ('k01_polynomial_ring',   'kill', 'k01_novel', 'polynomial_ring_attack'),
    ('k01_multi_exponent',    'kill', 'k01_novel', 'multi_exponent_attack'),
    ('k01_novel_combined',    'kill', 'k01_novel', 'novel_combined'),

    ('k01b_frobenius_walk',       'kill', 'k01_novel2', 'frobenius_walk'),
    ('k01b_frobenius_systematic', 'kill', 'k01_novel2', 'frobenius_systematic'),
    ('k01b_iterated_frobenius',   'kill', 'k01_novel2', 'iterated_frobenius'),
    ('k01b_cubic_frobenius',      'kill', 'k01_novel2', 'cubic_frobenius'),
    ('k01b_multi_ring',           'kill', 'k01_novel2', 'multi_ring_attack'),
    ('k01b_novel2_combined',      'kill', 'k01_novel2', 'novel2_combined'),

    ('k02_ultimate', 'kill', 'k02_ultimate', 'ultimate_factor'),

    ('k03_factor_algebraic',         'kill', 'k03_algebraic', 'factor_algebraic'),
    ('k03_factor_algebraic_modular', 'kill', 'k03_algebraic', 'factor_algebraic_modular'),
    ('k03_factor_algebraic_bitwise', 'kill', 'k03_algebraic', 'factor_algebraic_bitwise'),

    ('k03_coppersmith_pipeline', 'kill', 'k03_coppersmith', 'try_coppersmith_pipeline'),
]


WRAPPER_TEMPLATE = '''\
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "kills"))
sys.path.insert(0, _HERE)

from {module} import {func} as _impl

def factor(N):
    try:
        r = _impl(N)
    except Exception:
        return None
    if r is None:
        return None
    if isinstance(r, tuple) and len(r) == 2:
        return (int(r[0]), int(r[1]))
    if isinstance(r, list) and len(r) >= 2:
        return (int(r[0]), int(r[1]))
    return None
'''


def write_wrapper(name, module, func):
    path = os.path.join(HERE, f'_wrap_{name}.py')
    with open(path, 'w') as f:
        f.write(WRAPPER_TEMPLATE.format(module=module, func=func))
    return path


def cleanup_wrapper(name):
    path = os.path.join(HERE, f'_wrap_{name}.py')
    try:
        os.remove(path)
    except OSError:
        pass


def write_leaderboard(results, leaderboard_path, in_progress, n_total):
    ranked = sorted(
        [r for r in results if 'max_bits' in r],
        key=lambda r: (
            -r['max_bits'],
            0 if r['best_model'] == 'POLY' else (1 if r['best_model'] == 'SUBEXP' else 2),
            r['poly_k'] if r['poly_k'] < float('inf') else 9999,
            r['time_at_max'],
        ),
    )

    lines = []
    lines.append("# FactorCup Leaderboard")
    lines.append("")
    lines.append("Verified scores from `factorcup/score.py`. Every entry runs in a spawn-isolated subprocess with parent-side timing and `os.urandom`-seeded test cases.")
    lines.append("")
    lines.append("**Columns**")
    lines.append("- `max_bits`: largest bit size where at least 3 of 5 cases solved")
    lines.append("- `model`: best-fit scaling (POLY, EXP, SUBEXP=L[1/3])")
    lines.append("- `poly_k`: exponent under the polynomial fit (lower is better)")
    lines.append("- `time@max`: median time over the 5 cases at `max_bits`")
    lines.append("")
    if in_progress:
        lines.append(f"> **Run in progress** ({len(results)}/{n_total} entries scored). Table updates as each entry finishes.")
        lines.append("")
    lines.append("| # | Algorithm | max_bits | model | poly_k | R²_poly | R²_sub | time@max |")
    lines.append("|---|-----------|---------:|:-----:|-------:|--------:|-------:|---------:|")
    for rank, r in enumerate(ranked, 1):
        k_str = f"{r['poly_k']:.2f}" if r['poly_k'] < 500 else "—"
        rp = f"{r['poly_r2']:.3f}" if r['poly_r2'] > 0 else "—"
        rs = f"{r['subexp_r2']:.3f}" if r['subexp_r2'] > 0 else "—"
        t_str = f"{r['time_at_max']:.3f}s" if r['time_at_max'] < float('inf') else "—"
        flag = " ⚠" if r.get('flagged') else ""
        lines.append(f"| {rank} | `{r['name']}`{flag} | {r['max_bits']} | {r['best_model']} | {k_str} | {rp} | {rs} | {t_str} |")
    errored = [r for r in results if 'error' in r]
    if errored:
        lines.append("")
        lines.append("### Errored (failed to load or run)")
        lines.append("")
        for r in errored:
            lines.append(f"- `{r['name']}`: {r['error']}")
    lines.append("")
    lines.append(f"Anti-cheat: spawn isolation, parent-side timing, `os.urandom` test-case seeds. Parallelism: {PARALLELISM} concurrent entries on an 8-core host (timings may inflate slightly when multiple CPU-bound entries overlap).")
    lines.append("")

    with open(leaderboard_path, 'w') as f:
        f.write("\n".join(lines))


def main():
    leaderboard_path = os.path.join(HERE, 'LEADERBOARD.md')
    out_dir = os.path.join(HERE, 'leaderboard_results')
    os.makedirs(out_dir, exist_ok=True)

    # Wipe stale results
    for f in os.listdir(out_dir):
        if f.endswith('.json'):
            os.remove(os.path.join(out_dir, f))

    # Pre-write all wrappers
    for name, location, module, func in ENTRIES:
        if location == 'native':
            continue
        write_wrapper(name, module, func)

    # Build job list of (name, mod_name)
    jobs = []
    for name, location, module, func in ENTRIES:
        mod_name = 'entry' if location == 'native' else f'_wrap_{name}'
        jobs.append((name, mod_name))

    print(f"Total entries: {len(jobs)}")
    print(f"Parallelism: {PARALLELISM}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    log_path = os.path.join(HERE, 'leaderboard_run.log')
    log_fp = open(log_path, 'w')

    running = []   # list of (name, popen)
    job_iter = iter(jobs)
    pending = list(jobs)
    finished = 0
    results = []

    def launch_next():
        nonlocal pending
        if not pending:
            return None
        name, mod_name = pending.pop(0)
        cmd = [PY, os.path.join(HERE, '_score_one.py'), name, mod_name]
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=log_fp, stderr=log_fp)
        return (name, proc, time.time())

    # Prime initial pool
    while len(running) < PARALLELISM:
        nxt = launch_next()
        if nxt is None:
            break
        running.append(nxt)
        print(f"[launched] {nxt[0]}")

    # Poll loop
    while running:
        time.sleep(2.0)
        still = []
        for name, proc, t0 in running:
            if proc.poll() is None:
                still.append((name, proc, t0))
                continue
            # Done
            elapsed = time.time() - t0
            finished += 1
            json_path = os.path.join(out_dir, f'{name}.json')
            if os.path.exists(json_path):
                try:
                    with open(json_path) as fp:
                        r = json.load(fp)
                    results.append(r)
                    if 'max_bits' in r:
                        print(f"[done {finished}/{len(jobs)}] {name}: max_bits={r['max_bits']} model={r['best_model']} ({elapsed:.0f}s)")
                    else:
                        print(f"[err  {finished}/{len(jobs)}] {name}: {r.get('error', 'unknown')}")
                except Exception as e:
                    print(f"[parse-fail {finished}/{len(jobs)}] {name}: {e}")
                    results.append({'name': name, 'error': f'parse-fail: {e}'})
            else:
                print(f"[no-json {finished}/{len(jobs)}] {name} (exit {proc.returncode}, {elapsed:.0f}s)")
                results.append({'name': name, 'error': f'no json (exit {proc.returncode})'})

            write_leaderboard(results, leaderboard_path, in_progress=True, n_total=len(jobs))

            # Launch a replacement
            nxt = launch_next()
            if nxt is not None:
                still.append(nxt)
                print(f"[launched] {nxt[0]}")
        running = still

    # Cleanup wrappers
    for name, location, module, func in ENTRIES:
        if location != 'native':
            cleanup_wrapper(name)

    write_leaderboard(results, leaderboard_path, in_progress=False, n_total=len(jobs))
    log_fp.close()
    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
