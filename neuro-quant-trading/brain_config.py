"""
brain_config.py
---------------
Shared configuration: brain region definitions, trading scenario states,
and colour palette. Import this in every other module.

Neuroscience references:
  Lo & Repin (2002) – Psychophysiology of real-time financial risk processing
  Kuhnen & Knutson (2005) – Neural basis of financial risk taking
  Frydman et al. (2014) – Using neural data to test a theory of investor behaviour
  De Martino et al. (2006) – Frames, biases, and rational decision-making (amygdala)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Brain regions: MNI-approximate centroids (x, y, z) in mm
# ---------------------------------------------------------------------------
BRAIN_REGIONS = {
    # Region name : (x, y, z, radius_mm, label)
    "amygdala_L":          (-24, -4,  -18, 8,  "Amygdala\n(Loss Aversion)"),
    "amygdala_R":          ( 24, -4,  -18, 8,  "Amygdala\n(Loss Aversion)"),
    "ventral_striatum_L":  (-12,  8,   -8, 9,  "Ventral Striatum\n(Reward Anticipation)"),
    "ventral_striatum_R":  ( 12,  8,   -8, 9,  "Ventral Striatum\n(Reward Anticipation)"),
    "dlPFC_L":             (-36, 28,   36, 10, "dlPFC\n(Rule-Following)"),
    "dlPFC_R":             ( 36, 28,   36, 10, "dlPFC\n(Rule-Following)"),
    "anterior_insula_L":   (-38, 10,    2, 8,  "Anterior Insula\n(Risk / Disgust)"),
    "anterior_insula_R":   ( 38, 10,    2, 8,  "Anterior Insula\n(Risk / Disgust)"),
    "ACC":                 (  0, 24,   28, 9,  "ACC\n(Conflict Monitor)"),
    "vmPFC":               (  0, 48,  -10, 8,  "vmPFC\n(Value Signal)"),
    "OFC_L":               (-20, 36,  -14, 7,  "OFC\n(Expected Value)"),
    "OFC_R":               ( 20, 36,  -14, 7,  "OFC\n(Expected Value)"),
    "hippocampus_L":       (-28,-22,  -14, 8,  "Hippocampus\n(Memory Bias)"),
    "hippocampus_R":       ( 28,-22,  -14, 8,  "Hippocampus\n(Memory Bias)"),
}

# Colour per functional group (matplotlib RGBA or hex)
REGION_COLOURS = {
    "amygdala_L":         "#E84040",   # red   – fear/loss
    "amygdala_R":         "#E84040",
    "ventral_striatum_L": "#F5A623",   # amber – reward
    "ventral_striatum_R": "#F5A623",
    "dlPFC_L":            "#4A90D9",   # blue  – executive
    "dlPFC_R":            "#4A90D9",
    "anterior_insula_L":  "#9B59B6",   # purple– aversion
    "anterior_insula_R":  "#9B59B6",
    "ACC":                "#2ECC71",   # green – control
    "vmPFC":              "#1ABC9C",   # teal  – value
    "OFC_L":              "#F39C12",   # orange
    "OFC_R":              "#F39C12",
    "hippocampus_L":      "#BDC3C7",   # grey  – memory
    "hippocampus_R":      "#BDC3C7",
}

# ---------------------------------------------------------------------------
# Trading scenarios: which regions activate and how strongly (0–1)
# ---------------------------------------------------------------------------
TRADING_SCENARIOS = {
    "calm_market": {
        "label": "Calm Market",
        "subtitle": "Low vol, trend-following, plan execution",
        "activations": {
            "dlPFC_L": 0.85, "dlPFC_R": 0.85,
            "ACC": 0.60,
            "vmPFC": 0.55,
            "OFC_L": 0.50, "OFC_R": 0.50,
            "amygdala_L": 0.10, "amygdala_R": 0.10,
            "ventral_striatum_L": 0.30, "ventral_striatum_R": 0.30,
            "anterior_insula_L": 0.15, "anterior_insula_R": 0.15,
            "hippocampus_L": 0.40, "hippocampus_R": 0.40,
        },
        "pnl_drag_bps": 0,
        "colour": "#2ECC71",
    },
    "vol_spike": {
        "label": "Volatility Spike",
        "subtitle": "3-sigma GVZ event, position at risk",
        "activations": {
            "amygdala_L": 0.95, "amygdala_R": 0.95,
            "anterior_insula_L": 0.88, "anterior_insula_R": 0.88,
            "ACC": 0.75,
            "hippocampus_L": 0.70, "hippocampus_R": 0.70,
            "dlPFC_L": 0.35, "dlPFC_R": 0.35,
            "vmPFC": 0.20,
            "OFC_L": 0.25, "OFC_R": 0.25,
            "ventral_striatum_L": 0.15, "ventral_striatum_R": 0.15,
        },
        "pnl_drag_bps": -47,
        "colour": "#E84040",
    },
    "panic_sell": {
        "label": "Panic Exit",
        "subtitle": "Amygdala hijack — exits plan prematurely",
        "activations": {
            "amygdala_L": 1.00, "amygdala_R": 1.00,
            "anterior_insula_L": 0.95, "anterior_insula_R": 0.95,
            "hippocampus_L": 0.90, "hippocampus_R": 0.90,
            "ACC": 0.30,
            "dlPFC_L": 0.12, "dlPFC_R": 0.12,
            "vmPFC": 0.08,
            "OFC_L": 0.10, "OFC_R": 0.10,
            "ventral_striatum_L": 0.05, "ventral_striatum_R": 0.05,
        },
        "pnl_drag_bps": -112,
        "colour": "#C0392B",
    },
    "reward_chase": {
        "label": "FOMO / Reward Chase",
        "subtitle": "Oversize entry after winning streak",
        "activations": {
            "ventral_striatum_L": 1.00, "ventral_striatum_R": 1.00,
            "OFC_L": 0.80, "OFC_R": 0.80,
            "vmPFC": 0.75,
            "amygdala_L": 0.25, "amygdala_R": 0.25,
            "dlPFC_L": 0.20, "dlPFC_R": 0.20,
            "anterior_insula_L": 0.10, "anterior_insula_R": 0.10,
            "ACC": 0.18,
            "hippocampus_L": 0.35, "hippocampus_R": 0.35,
        },
        "pnl_drag_bps": -63,
        "colour": "#F5A623",
    },
    "recovery": {
        "label": "Controlled Recovery",
        "subtitle": "dlPFC re-engagement, back to process",
        "activations": {
            "dlPFC_L": 0.90, "dlPFC_R": 0.90,
            "ACC": 0.80,
            "vmPFC": 0.65,
            "OFC_L": 0.60, "OFC_R": 0.60,
            "amygdala_L": 0.20, "amygdala_R": 0.20,
            "anterior_insula_L": 0.30, "anterior_insula_R": 0.30,
            "ventral_striatum_L": 0.40, "ventral_striatum_R": 0.40,
            "hippocampus_L": 0.50, "hippocampus_R": 0.50,
        },
        "pnl_drag_bps": -8,
        "colour": "#4A90D9",
    },
}

# Algo has no activation — it operates on signals, not neuroendocrine state
ALGO_ACTIVATIONS = {k: 0.0 for k in BRAIN_REGIONS}

# ---------------------------------------------------------------------------
# PnL simulation parameters (Gold futures, 1 contract = $100/pt)
# ---------------------------------------------------------------------------
np.random.seed(42)
TRADING_DAYS = 252
BASE_EDGE_BPS = 18          # raw strategy edge per trade (bps)
TRADES_PER_DAY = 3
CONTRACT_VALUE = 100        # $/point for GC

# Scenario sequence for the animation (indices into TRADING_SCENARIOS)
SCENARIO_SEQUENCE = [
    "calm_market",
    "calm_market",
    "vol_spike",
    "panic_sell",
    "calm_market",
    "reward_chase",
    "vol_spike",
    "recovery",
    "calm_market",
    "calm_market",
]
