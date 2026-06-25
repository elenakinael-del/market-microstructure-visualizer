"""
run_all.py
----------
Runs all five scripts in sequence. Call from the project root:

    python run_all.py

Or run individual scripts:

    python 03_bias_cost_breakdown.py   # fast static poster (~5 s)
    python 05_interactive_brain_plotly.py  # interactive HTML (~10 s)
    python 01_brain_activation_atlas.py   # animation (~2 min)
    python 02_pnl_human_vs_algo.py        # animation (~1 min)
    python 04_master_animation.py         # master animation (~4 min)
"""

import subprocess, sys, os, time

SCRIPTS = [
    ("03_bias_cost_breakdown.py",       "Static bias cost poster (PNG/PDF)"),
    ("05_interactive_brain_plotly.py",  "Interactive Plotly brain (HTML)"),
    ("01_brain_activation_atlas.py",    "Brain activation animation (MP4)"),
    ("02_pnl_human_vs_algo.py",         "PnL comparison animation (MP4)"),
    ("04_master_animation.py",          "Master combined animation (MP4)"),
]

root = os.path.dirname(os.path.abspath(__file__))

for script, description in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  → {script}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(root, script)],
        cwd=root,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✓ Done in {elapsed:.1f}s")
    else:
        print(f"  ✗ Failed (exit {result.returncode})")

print("\nAll outputs saved to:  outputs/")
