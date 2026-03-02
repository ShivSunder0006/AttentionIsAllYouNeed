"""Positional Encoding — generate and visualise the sine/cosine PE matrix."""

import numpy as np
import plotly.graph_objects as go


def generate_pe_matrix(seq_len: int, d_model: int) -> np.ndarray:
    """
    Build the positional-encoding matrix using the formulas from
    "Attention Is All You Need":

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Returns an (seq_len × d_model) NumPy array.
    """
    positions = np.arange(seq_len)[:, np.newaxis]          # (seq_len, 1)
    dims = np.arange(d_model)[np.newaxis, :]               # (1, d_model)
    angles = positions / np.power(10_000, (2 * (dims // 2)) / d_model)

    pe = np.zeros_like(angles)
    pe[:, 0::2] = np.sin(angles[:, 0::2])  # even dims → sin
    pe[:, 1::2] = np.cos(angles[:, 1::2])  # odd  dims → cos
    return pe


def plot_pe_heatmap(pe_matrix: np.ndarray, seq_len: int, d_model: int) -> go.Figure:
    """Return a Plotly heatmap of the positional-encoding matrix."""
    fig = go.Figure(
        data=go.Heatmap(
            z=pe_matrix,
            x=[f"d={i}" for i in range(d_model)],
            y=[f"pos={i}" for i in range(seq_len)],
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Value", tickfont=dict(color="#E0E0E0")),
            hovertemplate="Position: %{y}<br>Dimension: %{x}<br>Value: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Positional Encoding  (seq_len={seq_len}, d_model={d_model})",
            font=dict(size=18, color="#E0E0E0"),
        ),
        xaxis=dict(
            title="Embedding Dimension",
            tickfont=dict(color="#B0B0B0"),
            titlefont=dict(color="#E0E0E0"),
            showgrid=False,
        ),
        yaxis=dict(
            title="Position in Sequence",
            tickfont=dict(color="#B0B0B0"),
            titlefont=dict(color="#E0E0E0"),
            autorange="reversed",
            showgrid=False,
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(l=60, r=30, t=50, b=50),
        height=max(400, seq_len * 6),
    )
    return fig
