"""Original Plotly diagrams and shared-scale velocity visualizations."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE = "#176B87"
ORANGE = "#C25728"
INK = "#163340"


def _style(fig, height=370):
    fig.update_layout(
        height=height, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color=INK, size=13),
        margin=dict(l=55, r=25, t=45, b=55),
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E7EFF2", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E7EFF2", zeroline=False)
    return fig


def profile_figure(fluids, trajectories, frame):
    fig = go.Figure()
    for fluid, trajectory, color in zip(fluids, trajectories, [BLUE, ORANGE]):
        y, velocity, _ = trajectory
        short_name = ("Baseline" if fluid.label == "Baseline" else "Changed case") if len(fluids) == 2 else "Fluid"
        fig.add_trace(go.Scatter(
            x=velocity[frame], y=y * 1000, mode="lines", name=short_name,
            line=dict(color=color, width=3),
            hovertemplate="u = %{x:.5f} m/s<br>y = %{y:.3f} mm<extra>%{fullData.name}</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=fluid.U * y / fluid.H, y=y * 1000, mode="lines",
            name=f"{short_name}: steady", line=dict(color=color, dash="dash", width=1.6),
            hovertemplate="Steady u = %{x:.5f} m/s<br>y = %{y:.3f} mm<extra></extra>",
        ))
    fig.update_xaxes(title="Horizontal velocity u (m/s)", range=[0, 1.05 * max(f.U for f in fluids)])
    fig.update_yaxes(title="Position across gap y (mm)", range=[0, 1.06 * max(f.H for f in fluids) * 1000])
    return _style(fig, 400)


def arrows_figure(fluids, trajectories, frame):
    count = len(fluids)
    # Stacked comparisons keep each arrow panel readable at phone widths.
    fig = make_subplots(rows=count, cols=1, subplot_titles=[f.label for f in fluids], vertical_spacing=0.20 if count == 2 else 0.0)
    max_speed = max(f.U for f in fluids)
    max_gap_mm = max(f.H for f in fluids) * 1000
    for column, (fluid, trajectory, color) in enumerate(zip(fluids, trajectories, [BLUE, ORANGE]), 1):
        y, velocity, _ = trajectory
        for plate, text in [(0, "Fixed plate"), (fluid.H * 1000, "Moving plate →")]:
            fig.add_shape(type="line", x0=0, x1=1.05, y0=plate, y1=plate,
                          line=dict(color=INK, width=4), row=column, col=1)
            fig.add_trace(go.Scatter(x=[0.5], y=[plate], text=[text], mode="text",
                                    textposition="bottom center" if plate == 0 else "top center",
                                    showlegend=False, hoverinfo="skip"), row=column, col=1)
        for index in np.linspace(0, len(y) - 1, 11).astype(int):
            yy = y[index] * 1000
            arrow_length = 0.88 * velocity[frame, index] / max_speed
            if arrow_length > 1e-6:
                xref = "x" if column == 1 else f"x{column}"
                yref = "y" if column == 1 else f"y{column}"
                fig.add_annotation(x=0.08 + arrow_length, y=yy, ax=0.08, ay=yy,
                                   xref=xref, yref=yref, axref=xref, ayref=yref,
                                   text="", showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor=color)
            fig.add_trace(go.Scatter(x=[0.08], y=[yy], mode="markers", marker=dict(color=color, size=4),
                                    showlegend=False, hoverinfo="skip"), row=column, col=1)
        fig.update_xaxes(range=[0, 1.07], showticklabels=False, showgrid=False,
                         title="Arrow length ∝ speed", row=column, col=1)
        fig.update_yaxes(range=[-0.10 * max_gap_mm, 1.12 * max_gap_mm],
                         title="y (mm)", row=column, col=1)
    return _style(fig, 650 if count == 2 else 375)


def force_diagram():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.2, x1=0.8, y0=0.35, y1=0.65,
                  fillcolor="#DBEEF4", line=dict(color=BLUE, width=2))
    fig.add_annotation(x=0.5, y=0.5, text="Thin layer: mass = ρ A Δy", showarrow=False)
    fig.add_annotation(x=0.83, y=0.67, ax=0.25, ay=0.67, xref="x", yref="y", axref="x", ayref="y",
                       text="", showarrow=True, arrowhead=3, arrowcolor=BLUE, arrowwidth=3)
    fig.add_annotation(x=0.17, y=0.33, ax=0.7, ay=0.33, xref="x", yref="y", axref="x", ayref="y",
                       text="", showarrow=True, arrowhead=3, arrowcolor=ORANGE, arrowwidth=3)
    fig.add_annotation(x=0.5, y=0.82, text="Upper face at y + Δy:<br>A τ(y + Δy) →", showarrow=False)
    fig.add_annotation(x=0.5, y=0.18, text="Lower face at y:<br>← A τ(y)", showarrow=False)
    fig.add_annotation(x=0.5, y=0.03, text="Horizontal forces.<br>Δy measures thickness.", showarrow=False)
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False)
    return _style(fig, 280)


def cylinder_diagram():
    fig = go.Figure()
    fig.add_shape(type="circle", x0=-1, x1=1, y0=-1, y1=1, fillcolor="#DBEEF4", line=dict(color=BLUE, width=3))
    fig.add_shape(type="circle", x0=-0.72, x1=0.72, y0=-0.72, y1=0.72,
                  fillcolor="#F4F1E9", line=dict(color=INK, width=3))
    fig.add_trace(go.Scatter(x=[0, .72], y=[0, 0], mode="lines", line=dict(color=INK), showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, 1], mode="lines", line=dict(color=INK), showlegend=False))
    fig.add_annotation(x=.36, y=-.13, text="Rᵢ", showarrow=False)
    fig.add_annotation(x=.12, y=.52, text="Rₒ", showarrow=False)
    fig.add_annotation(x=-.24, y=-.45, text="Stationary<br>inner cylinder", showarrow=False)
    fig.add_annotation(x=.88, y=.18, text="Fluid", showarrow=False)
    theta = np.linspace(.1, 1.2, 40)
    fig.add_trace(go.Scatter(x=1.12 * np.cos(theta), y=1.12 * np.sin(theta), mode="lines",
                            line=dict(color=ORANGE, width=3), showlegend=False, hoverinfo="skip"))
    fig.add_annotation(x=1.12*np.cos(1.2), y=1.12*np.sin(1.2),
                       ax=1.12*np.cos(1.05), ay=1.12*np.sin(1.05), xref="x", yref="y", axref="x", ayref="y",
                       text="", showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=ORANGE)
    fig.add_annotation(x=0, y=1.48, text="Rotating outer cup ↺", showarrow=False)
    fig.add_annotation(x=0, y=-1.35, text="Gap H = Rₒ − Rᵢ (schematic, not to scale)", showarrow=False)
    fig.update_xaxes(range=[-1.7, 1.7], visible=False)
    fig.update_yaxes(range=[-1.65, 1.65], visible=False, scaleanchor="x", scaleratio=1)
    return _style(fig, 390)
