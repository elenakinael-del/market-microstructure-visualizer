import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("[NEURAL GRID] Initializing Synaptic Connection Arrays...")

# --------------------------------------------------------
# 1. GENERATE GEOMETRIES & SYNAPSE PAIRS
# --------------------------------------------------------
n_frames = 80
n_points = 200

np.random.seed(42)
brain_x = np.random.normal(0, 0.6, n_points)
brain_y = np.random.normal(0, 0.9, n_points)
brain_z = np.random.normal(0, 0.5, n_points)
brain_x = np.where(brain_x > 0, brain_x + 0.15, brain_x - 0.15) 

# Create localized structural connections (Synapses) between random close nodes
synapse_pairs = []
for i in range(n_points):
    # Connect each node to 2 nearby neighbors to create an organic net
    distances = np.sqrt((brain_x - brain_x[i])**2 + (brain_y - brain_y[i])**2 + (brain_z - brain_z[i])**2)
    nearest_indices = np.argsort(distances)[1:3] 
    for idx in nearest_indices:
        if (idx, i) not in synapse_pairs:
            synapse_pairs.append((i, idx))

# Generate Algo Grid
grid_range = np.linspace(-1, 1, 6)
algo_x, algo_y, algo_z = np.meshgrid(grid_range, grid_range, grid_range)
algo_x, algo_y, algo_z = algo_x.flatten(), algo_y.flatten(), algo_z.flatten()
n_algo = len(algo_x)

# Connect Algos linearly along the X-axis lanes
algo_pairs = []
for i in range(n_algo - 1):
    if np.abs(algo_x[i] - algo_x[i+1]) < 0.5:
        algo_pairs.append((i, i+1))

# Helper to flatten connection coordinates for Plotly's line renderer
def build_line_coords(x_arr, y_arr, z_arr, pairs):
    lx, ly, lz = [], [], []
    for p1, p2 in pairs:
        lx.extend([x_arr[p1], x_arr[p2], None])
        ly.extend([y_arr[p1], y_arr[p2], None])
        lz.extend([z_arr[p1], z_arr[p2], None])
    return lx, ly, lz

brain_lx, brain_ly, brain_lz = build_line_coords(brain_x, brain_y, brain_z, synapse_pairs)
algo_lx, algo_ly, algo_lz = build_line_coords(algo_x, algo_y, algo_z, algo_pairs)

# --------------------------------------------------------
# 2. FRAME GENERATION SYSTEM (ANIMATED BIOMETRICS)
# --------------------------------------------------------
frames = []
t_space = np.linspace(0, 4 * np.pi, n_frames)
human_pnl, algo_pnl = 100, 100

for i, t in enumerate(t_space):
    node_states = np.zeros(n_points)
    
    # SYSTEM STATE LOGIC
    if 20 <= i < 45:   # AMYGDALA FLARE PHASE
        amygdala_mask = (np.abs(brain_x) < 0.4) & (np.abs(brain_y) < 0.4) & (brain_z < 0)
        node_states[amygdala_mask] = 1.0
        human_pnl -= np.random.uniform(2, 5)
        algo_pnl += np.random.uniform(0.5, 1.5)
        phase_text = "CRITICAL METRIC FLUIDITY: Amygdala Overdrive (Loss Aversion Spikes Execution Latency)"
        synapse_color = "#ff3300" # Angry crimson synapses
    elif i >= 45:      # FRONTAL CORTEX PIVOT
        cortex_mask = (brain_y > 0.4) & (brain_z > 0.1)
        node_states[cortex_mask] = 2.0
        human_pnl += np.random.uniform(4, 8)
        algo_pnl -= np.random.uniform(1, 3)
        phase_text = "REGIME BREAK DETECTED: Frontal Cortex Dominance (Intuitive Edge Captures Alpha)"
        synapse_color = "#ffaa00" # Golden intuition synapses
    else:              # STABLE BASELINE
        human_pnl += np.random.uniform(-0.5, 1)
        algo_pnl += np.random.uniform(0.5, 1.8)
        phase_text = "EQUILIBRIUM STATE: Algorithmic Arbitrage Harvesting Structured Variance"
        synapse_color = "#005577" # Deep ambient ocean-blue connections

    # Set individual point colors based on brain activity state
    node_colors = []
    for state in node_states:
        if state == 1.0: node_colors.append('#ff0055')   # Flashing Hot Pink
        elif state == 2.0: node_colors.append('#ffb700') # Neon Amber Gold
        else: node_colors.append('#00ffff')              # Electric Cyan Core

    # Dynamic flashing for the algo's computational matrix lines
    algo_line_color = '#00ff66' if np.sin(t * 2) > 0 else '#220066'
    status_msg = f"<b>Frame: {i} | {phase_text}<br>Human Balance: ${human_pnl:.2f} | Algo Balance: ${algo_pnl:.2f}</b>"

    frames.append(go.Frame(
        data=[
            # Left: Brain Nodes & Synapse Paths
            go.Scatter3d(x=brain_x, y=brain_y, z=brain_z, mode='markers', marker=dict(size=4.5, color=node_colors)),
            go.Scatter3d(x=brain_lx, y=brain_ly, z=brain_lz, mode='lines', line=dict(color=synapse_color, width=1.5), opacity=0.4),
            # Right: Algo Cores & Grid Paths
            go.Scatter3d(x=algo_x, y=algo_y, z=algo_z, mode='markers', marker=dict(size=3, color='#4411aa')),
            go.Scatter3d(x=algo_lx, y=algo_ly, z=algo_lz, mode='lines', line=dict(color=algo_line_color, width=1), opacity=0.5)
        ],
        layout=go.Layout(
            annotations=[dict(
                text=status_msg, x=0.5, y=1.15, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="#ffffff", family="Courier New")
            )]
        ),
        name=f'f_{i}'
    ))

# --------------------------------------------------------
# 3. CONFIGURE TARGET SUBPLOTS
# --------------------------------------------------------
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=(
        '<b>DISCRETIONARY BIOMETRIC PROFILE</b><br>Neural Synaptic Activation Grid', 
        '<b>QUANTUM ALGORITHMIC MATRIX</b><br>Rigid High-Frequency Data Lanes'
    )
)

# Insert baseline structures into Plotly framework
fig.add_trace(go.Scatter3d(x=brain_x, y=brain_y, z=brain_z, mode='markers', marker=dict(size=4.5, color='#00ffff')), row=1, col=1)
fig.add_trace(go.Scatter3d(x=brain_lx, y=brain_ly, z=brain_lz, mode='lines', line=dict(color='#005577', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter3d(x=algo_x, y=algo_y, z=algo_z, mode='markers', marker=dict(size=3, color='#4411aa')), row=1, col=2)
fig.add_trace(go.Scatter3d(x=algo_lx, y=algo_ly, z=algo_lz, mode='lines', line=dict(color='#220066', width=1)), row=1, col=2)

# Global Dark Void Layout Theme
fig.update_layout(
    title=dict(
        text="NEUROFINANCE SYNAPSE SIMULATION V2",
        font=dict(size=18, color="#ffffff", family="Courier New"),
        y=0.96, x=0.5, xanchor='center'
    ),
    paper_bgcolor='#000000', # Pitch Black
    plot_bgcolor='#000000',
    showlegend=False,
    
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    scene2=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    
    updatemenus=[dict(
        type="buttons",
        buttons=[
            dict(label="► Stream Synapse Network", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
            dict(label="▌▌ Halt Network", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
        ],
        direction="left", pad={"r": 10, "t": 10}, showactive=True,
        x=0.5, xanchor="center", y=-0.05, yanchor="top"
    )]
)

for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color='#ffffff', family="Courier New", size=13)

fig.frames = frames

print("[NEURAL GRID] System checks clear. Opening holographic canvas portal...")
fig.show()