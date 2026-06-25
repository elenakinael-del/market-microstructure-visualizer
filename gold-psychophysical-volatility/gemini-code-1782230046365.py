import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("[QUANTUM HYPERVISOR] Compiling Dual-Core Neural vs. Matrix Engine...")

# --------------------------------------------------------
# 1. GENERATE SCI-FI GEOMETRIES (BRAIN & LATTICE)
# --------------------------------------------------------
n_frames = 80
n_points = 300

# Generate a rough dual-hemisphere point-cloud (The Brain)
np.random.seed(42)
brain_x = np.random.normal(0, 0.6, n_points)
brain_y = np.random.normal(0, 0.9, n_points)
brain_z = np.random.normal(0, 0.5, n_points)
# Sculpt it loosely into two lobes
brain_x = np.where(brain_x > 0, brain_x + 0.2, brain_x - 0.2) 

# Generate a rigid crystal lattice (The Algo)
grid_range = np.linspace(-1, 1, 7)
algo_x, algo_y, algo_z = np.meshgrid(grid_range, grid_range, grid_range)
algo_x = algo_x.flatten()
algo_y = algo_y.flatten()
algo_z = algo_z.flatten()
n_algo_points = len(algo_x)

# --------------------------------------------------------
# 2. BUILD THE ANIMATION SCENARIO (THE FLUCTUATING PNL)
# --------------------------------------------------------
frames = []
t_space = np.linspace(0, 4 * np.pi, n_frames)

# Tracking PnL over time
human_pnl = 100
algo_pnl = 100

for i, t in enumerate(t_space):
    colors = np.zeros(n_points)
    
    if 20 <= i < 45: # MARKET PANIC PHASE
        amygdala_mask = (np.abs(brain_x) < 0.4) & (np.abs(brain_y) < 0.4) & (brain_z < 0)
        colors[amygdala_mask] = 1.0  
        human_pnl -= np.random.uniform(2, 6) 
        algo_pnl += np.random.uniform(0.5, 1.5) 
        phase_text = "MARKET ANOMALY: Human Amygdala Flare (Loss Aversion Traps Discretionary)"
    elif i >= 45: # SYSTEMIC BLACK SWAN BREAKOUT
        cortex_mask = (brain_y > 0.4) & (brain_z > 0.2)
        colors[cortex_mask] = 2.0  
        human_pnl += np.random.uniform(4, 9) 
        algo_pnl -= np.random.uniform(1, 3)  
        phase_text = "REGIME SHIFT: Human Intuition Capitalizes on Out-of-Sample Volatility"
    else: # NORMAL MATRIX CONDITIONS
        human_pnl += np.random.uniform(-1, 1.5)
        algo_pnl += np.random.uniform(0.5, 2.0)
        phase_text = "STANDARD EFFICIENCY: Algo Core Accumulating Alpha via Micro-Arbitrage"

    brain_colors = []
    for c in colors:
        if c == 1.0: brain_colors.append('#ff3300') 
        elif c == 2.0: brain_colors.append('#ffaa00') 
        else: brain_colors.append('#00e5ff') 

    algo_colors = ['#00ff66' if np.sin(t + ax) > 0.7 else '#220066' for ax in algo_x]
    status_annotation = f"Frame: {i} | {phase_text}<br>Human Equity: ${human_pnl:.2f} | Algo Equity: ${algo_pnl:.2f}"

    frames.append(go.Frame(
        data=[
            go.Scatter3d(
                x=brain_x, y=brain_y, z=brain_z,
                mode='markers',
                marker=dict(size=4 + np.sin(t)*1.5, color=brain_colors, opacity=0.7)
            ),
            go.Scatter3d(
                x=algo_x, y=algo_y, z=algo_z,
                mode='markers',
                marker=dict(size=3.5, color=algo_colors, opacity=0.8)
            )
        ],
        layout=go.Layout(
            annotations=[dict(
                text=status_annotation,
                x=0.5, y=1.15, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="#ffffff", family="Courier New")
            )]
        ),
        name=f'sim_{i}'
    ))

# --------------------------------------------------------
# 3. INITIAL HOUSING DASHBOARD CONFIGURATION
# --------------------------------------------------------
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=(
        '<b>HUMAN DISCRETIONARY CORE</b><br>Neural Cognitive Node Map', 
        '<b>ALGORITHMIC QUANTUM MATRIX</b><br>Linear Execution Lattice'
    )
)

fig.add_trace(go.Scatter3d(
    x=brain_x, y=brain_y, z=brain_z, mode='markers',
    name='Cognitive Density',
    marker=dict(size=5, color='#00e5ff', opacity=0.7)
), row=1, col=1)

fig.add_trace(go.Scatter3d(
    x=algo_x, y=algo_y, z=algo_z, mode='markers',
    name='Execution Sub-nodes',
    marker=dict(size=3.5, color='#220066', opacity=0.8)
), row=1, col=2)

# Global Layout Configuration (FIXED: Removed top-level backgroundcolor inside scenes)
fig.update_layout(
    title=dict(
        text="COGNITIVE SYNERGY SIMULATION: HUMAN NERVE VS MACHINE PATTERN",
        font=dict(size=18, color="#ffffff", family="Courier New"),
        y=0.95, x=0.5, xanchor='center'
    ),
    paper_bgcolor='#020206', 
    plot_bgcolor='#020206',
    showlegend=False,
    
    scene=dict(
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False), 
        zaxis=dict(visible=False)
    ),
    scene2=dict(
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False), 
        zaxis=dict(visible=False)
    ),
    
    updatemenus=[dict(
        type="buttons",
        buttons=[
            dict(label="► Initialize Core Streaming",
                 method="animate",
                 args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
            dict(label="▌▌ Freeze Core",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
        ],
        direction="left", pad={"r": 10, "t": 10}, showactive=True,
        x=0.5, xanchor="center", y=-0.05, yanchor="top"
    )]
)

for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color='#ffffff', family="Courier New", size=13)

fig.frames = frames

print("[QUANTUM HYPERVISOR] Bug corrected. Launching UI portal...")
fig.show()