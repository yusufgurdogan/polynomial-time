#!/usr/bin/env python3
"""
Score a single entry. Used by run_leaderboard.py to parallelize.

    python _score_one.py <display_name> <wrapper_module>

Writes JSON result to results/<display_name>.json.
"""

import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from score import score_algorithm, SCORING_BIT_SIZES


def main():
    if len(sys.argv) != 3:
        print("usage: _score_one.py <name> <module>")
        sys.exit(2)

    name = sys.argv[1]
    module = sys.argv[2]

    out_dir = os.path.join(HERE, 'leaderboard_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}.json')

    t0 = time.time()
    try:
        r = score_algorithm(name, module, SCORING_BIT_SIZES, quiet=True)
        r.pop('details', None)
        elapsed = time.time() - t0
        r['_runtime'] = elapsed
        with open(out_path, 'w') as f:
            json.dump(r, f)
        print(f"[done] {name}: max_bits={r['max_bits']} model={r['best_model']} ({elapsed:.0f}s)")
    except Exception as e:
        with open(out_path, 'w') as f:
            json.dump({'name': name, 'error': str(e)}, f)
        print(f"[fail] {name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
