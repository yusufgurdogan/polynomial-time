#!/usr/bin/env python3
"""Re-render LEADERBOARD.md from the JSON results dir without re-running."""

import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))


def render(results, leaderboard_path):
    alive = [r for r in results if r.get('max_bits', 0) > 0]
    dead  = [r for r in results if r.get('max_bits', 0) == 0 and 'error' not in r]
    errored = [r for r in results if 'error' in r]

    alive_ranked = sorted(alive, key=lambda r: (
        -r['max_bits'],
        0 if r['best_model'] == 'POLY' else (1 if r['best_model'] == 'SUBEXP' else 2),
        r['poly_k'] if r['poly_k'] < float('inf') else 9999,
        r['time_at_max'],
    ))

    lines = []
    lines.append("# FactorCup Leaderboard")
    lines.append("")
    lines.append("Verified scores from `factorcup/score.py` against every standalone factoring function in this repo. Each entry runs in a spawn-isolated subprocess with parent-side timing and `os.urandom`-seeded test cases. The scorer walks bit sizes 32 → 1024 with 5 random balanced semiprimes per size and a 60s per-case timeout, declaring an entry dead at the first bit size where it solves fewer than 3 of 5.")
    lines.append("")
    lines.append("**Columns**")
    lines.append("- `max_bits`: largest bit size where at least 3 of 5 cases solved")
    lines.append("- `model`: best-fit scaling (POLY, SUBEXP=L[1/3], EXP), chosen by R² on log-log regression")
    lines.append("- `R²`: goodness-of-fit for the chosen model")
    lines.append("- `poly_k`: exponent under the polynomial fit when meaningful")
    lines.append("- `time@max`: median time on the 5 cases at `max_bits`")
    lines.append("")
    lines.append("**Caveat on model selection.** With balanced semiprimes capped at 160 bits in our run, the fits across POLY/SUBEXP/EXP are often within a few percent of each other. Treat the `model` column as suggestive, not authoritative — `max_bits` and `time@max` are the load-bearing numbers.")
    lines.append("")
    lines.append("## Live entries")
    lines.append("")
    lines.append("| # | Algorithm | max_bits | model | R² | poly_k | time@max |")
    lines.append("|---|-----------|---------:|:-----:|---:|-------:|---------:|")
    for rank, r in enumerate(alive_ranked, 1):
        model = r['best_model']
        if model == 'POLY':
            r2 = r['poly_r2']
        elif model == 'EXP':
            r2 = r['exp_r2']
        else:
            r2 = r['subexp_r2']
        r2_str = f"{r2:.3f}" if r2 > 0 else "—"
        k_str = f"{r['poly_k']:.2f}" if r['poly_k'] < 500 else "—"
        t_str = f"{r['time_at_max']:.2f}s" if r['time_at_max'] < float('inf') else "—"
        flag = " ⚠" if r.get('flagged') else ""
        lines.append(f"| {rank} | `{r['name']}`{flag} | {r['max_bits']} | {model} | {r2_str} | {k_str} | {t_str} |")

    if dead:
        lines.append("")
        lines.append("## Dead at 32 bits")
        lines.append("")
        lines.append("These functions failed to solve 3 of 5 random 32-bit balanced semiprimes within 60s each. Most are research probes — they target specific structure (Coppersmith hints, partial info, smooth orders) that random semiprimes do not have.")
        lines.append("")
        for r in sorted(dead, key=lambda x: x['name']):
            lines.append(f"- `{r['name']}`")

    if errored:
        lines.append("")
        lines.append("## Errored")
        lines.append("")
        for r in errored:
            lines.append(f"- `{r['name']}`: {r['error']}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**How to reproduce.** From `factorcup/`: `pip install -r requirements.txt && python run_leaderboard.py`. Runs ~30 minutes wall-clock with 4-way parallelism on an 8-core box.")
    lines.append("")
    lines.append("**Anti-cheat.** Spawn isolation (no shared parent memory), parent-side `time.monotonic` (immune to child clock manipulation), `os.urandom` test-case seeds (unpredictable per run), and a `<50ms at 64+ bits` suspicion flag for impossibly fast solves.")
    lines.append("")

    with open(leaderboard_path, 'w') as f:
        f.write("\n".join(lines))


def main():
    out_dir = os.path.join(HERE, 'leaderboard_results')
    results = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.json'):
            with open(os.path.join(out_dir, f)) as fp:
                results.append(json.load(fp))
    leaderboard_path = os.path.join(HERE, 'LEADERBOARD.md')
    render(results, leaderboard_path)
    print(f"Wrote {leaderboard_path} with {len(results)} entries.")


if __name__ == "__main__":
    main()
