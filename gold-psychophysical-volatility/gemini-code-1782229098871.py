import numpy as np
import plotly.graph_objects as go

print("[QUANTUM CORE] Initializing Hyper-Dimensional Galactic Engine...")

# --------------------------------------------------------
# 1. GENERATE THE QUANTUM SPACE-TIME DATA (ANIMATED)
# --------------------------------------------------------
n_frames = 60
particles_per_system = 200

# Time vector for simulation
t = np.linspace(0, 4 * np.pi, n_frames)

frames = []

# Generate time-stamped positions for both systems
for i, ti in enumerate(t):
    # --- ALGORITHMIC SYSTEM (The Stable Quantum Lattice) ---
    # Algos move in tight, mathematically perfect, synchronized helical orbits
    algo_theta = np.linspace(0, 2 * np.pi, particles_per_system) + ti
    algo_x = 2 * np.cos(algo_theta)
    algo_y = 2 * np.sin(algo_theta)
    algo_z = np.linspace(-1, 1, particles_per_system) + 0.1 * np.sin(ti)
    
    # --- HUMAN DISCRETIONARY SYSTEM (The Chaotic Comet) ---
    # Humans drift organically, then experience sudden "neuro-shocks" (market panics)
    human_theta = np.linspace(0, 4 * np.pi, particles_per_system) + (ti * 0.5)
    # Add non-linear surge/panic mechanics to the human coordinates
    panic_factor = np.sin(ti * 1.5) * 1.8
    human_x = (3 + panic_factor) * np.cos(human_theta) + np.random.normal(0, 0.3, particles_per_system)
    human_y = (3 + panic_factor) * np.sin(human_theta) + np.random.normal(0, 0.3, particles_per_system)
    human_z = np.sin(human_theta * 0.5) * 3 + (ti * 0.2)

    # Compile this timestamp into an animation frame
    frames.append(go.Frame(
        data=[
            # Trace 0: Algos
            go.Scatter3d(
                x=algo_x, y=algo_y, z=algo_z,
                mode='markers',
                marker=dict(size=3, color='#00ffcc', opacity=0.7)
            ),
            # Trace 1: Humans
            go.Scatter3d(
                x=human_x, y=human_y, z=human_z,
                mode='markers',
                marker=dict(size=4.5, color='#ff0055', opacity=0.8)
            )
        ],
        name=f'frame_{i}'
    ))

# --------------------------------------------------------
# 2. BUILD THE INITIAL GALACTIC VIEW
# --------------------------------------------------------
fig = go.Figure(
    data=[
        # Initial Frame Algo State
        go.Scatter3d(
            x=frames[0].data[0].x, y=frames[0].data[0].y, z=frames[0].data[0].z,
            mode='markers',
            name='Algorithmic Matrix (High-Freq Lattice)',
            marker=dict(
                size=3,
                color='#00ffcc',
                colorscale=[[0, '#00ffcc'], [1, '#0033aa']],
                line=dict(width=0)
            )
        ),
        # Initial Frame Human State
        go.Scatter3d(
            x=frames[0].data[1].x, y=frames[0].data[1].y, z=frames[0].data[1].z,
            mode='markers',
            name='Human Discretionary (Neuro-Erratic Comet)',
            marker=dict(
                size=5,
                color='#ff0055',
                colorscale=[[0, '#ff0055'], [1, '#ffaa00']],
                line=dict(color='#ffffff', width=0.5)
            )
        )
    ],
    layout=go.Layout(
        title=dict(
            text="QUANTUM GALACTIC PHASE-SPACE: ALGO VS DISCRETIONARY",
            font=dict(size=18, color="#ffffff", family="Courier New")
        ),
        paper_bgcolor='#03030d', # Deep space black
        plot_bgcolor='#03030d',
        showlegend=True,
        legend=dict(font=dict(color="#ffffff"), bgcolor="rgba(0,0,0,0)"),
        
        # Cosmic Camera Layout & Dark Grid configurations
        scene=dict(
            xaxis=dict(title=dict(text='Momentum Vectors', font=dict(color='#6666ff')), 
                       gridcolor='#111133', backgroundcolor='#03030d', zerolinecolor='#222255'),
            yaxis=dict(title=dict(text='Order Book Depth', font=dict(color='#6666ff')), 
                       gridcolor='#111133', backgroundcolor='#03030d', zerolinecolor='#222255'),
            zaxis=dict(title=dict(text='Cognitive Entropy', font=dict(color='#6666ff')), 
                       gridcolor='#111133', backgroundcolor='#03030d', zerolinecolor='#222255')),
        
        # Interactive Controls for playing the "Video"
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="► Play Simulation",
                     method="animate",
                     args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                dict(label="▌▌ Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=True,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )]
    ),
    frames=frames
)

# Render Quantum Engine
print("[QUANTUM CORE] Simulation Ready. Launching interactive continuum...")
fig.show()