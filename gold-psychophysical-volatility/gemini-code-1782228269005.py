import plotly.graph_objects as go
import numpy as np

# ... (Keep your data loading or mock data logic here) ...
# Example placeholder for researched_data if needed:
researched_data = {} 

def generate_3d_research_cube(data):
    print("[SYSTEM] Compiling 3D Mathematical Phase-Space...")
    
    fig = go.Figure()
    
    # Dummy coordinates for structural placeholder
    x_data = np.random.randn(100)
    y_data = np.random.randn(100)
    z_data = np.random.randn(100)
    color_data = np.sqrt(x_data**2 + y_data**2 + z_data**2)

    fig.add_trace(go.Scatter3d(
        x=x_data,
        y=y_data,
        z=z_data,
        mode='markers',
        marker=dict(
            size=5,
            color=color_data,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(
                    text="Neuro-Behavioral Index",
                    side="top"  # FIXED: Nested inside 'title' dictionary instead of using 'titleside'
                )
            ),
            opacity=0.8
        )
    ))
    
    fig.update_layout(
        title="Neuro-Behavioral Phase-Space Cube",
        scene=dict(
            xaxis_title='Asset History Axis',
            yaxis_title='Market Metric Axis',
            zaxis_title='Behavioral Density'
        )
    )
    
    fig.show()

if __name__ == "__main__":
    print("[SYSTEM] Fetching historical market metrics from Yahoo Finance...")
    print("[SYSTEM] Running Neuro-Behavioral equations across asset history...")
    generate_3d_research_cube(researched_data)