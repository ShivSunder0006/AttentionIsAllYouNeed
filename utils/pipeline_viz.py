"""Pipeline Visualisation — show how a sentence flows through the Transformer."""

from __future__ import annotations

import functools
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.attention import load_model


# ---------------------------------------------------------------------------
# Step-by-step pipeline extraction
# ---------------------------------------------------------------------------

def run_pipeline(sentence: str, model_name: str = "bert-base-uncased") -> dict:
    """
    Run *sentence* through BERT and capture intermediate representations.
    """
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
        )

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    hidden_states = outputs.hidden_states
    attentions = outputs.attentions

    embeddings = hidden_states[0].squeeze(0).numpy()
    layer_outs = [hs.squeeze(0).numpy() for hs in hidden_states[1:]]
    attn_mats = [a.squeeze(0).numpy() for a in attentions]

    final = hidden_states[-1].squeeze(0)
    logits_norm = torch.linalg.norm(final, dim=-1).numpy()

    return {
        "tokens": tokens,
        "input_ids": inputs["input_ids"][0].numpy(),
        "embeddings": embeddings,
        "layer_outs": layer_outs,
        "attentions": attn_mats,
        "logits_norm": logits_norm,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_PURPLE = [
    [0.0, "#0E1117"],
    [0.25, "#1A1A2E"],
    [0.5, "#6C63FF"],
    [0.75, "#A78BFA"],
    [1.0, "#E0CFFC"],
]

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(color="#E0E0E0"),
    margin=dict(l=60, r=30, t=50, b=50),
)


def _styled(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_LAYOUT_DEFAULTS)
    return fig


def plot_tokenization(tokens: list[str], input_ids: np.ndarray) -> go.Figure:
    colors = [f"hsl({(i * 37) % 360}, 60%, 45%)" for i in range(len(tokens))]
    fig = go.Figure(
        go.Bar(
            x=[1] * len(tokens),
            y=[f"{tok}  (id {tid})" for tok, tid in zip(tokens, input_ids)],
            orientation="h",
            marker_color=colors,
            hovertemplate="Token: %{y}<extra></extra>",
            text=tokens,
            textposition="inside",
            textfont=dict(color="white", size=13),
        )
    )
    fig.update_layout(
        title=dict(text="Step 1 · Tokenization", font=dict(size=16)),
        xaxis=dict(visible=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#B0B0B0")),
        height=max(250, len(tokens) * 36),
        showlegend=False,
    )
    return _styled(fig)


def plot_embedding_heatmap(tokens: list[str], embeddings: np.ndarray) -> go.Figure:
    show_dims = min(embeddings.shape[1], 64)
    fig = go.Figure(
        go.Heatmap(
            z=embeddings[:, :show_dims],
            x=[str(d) for d in range(show_dims)],
            y=tokens,
            colorscale="RdBu_r",
            hovertemplate="Token: %{y}<br>Dim %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Step 2 · Token + Positional Embeddings  (first 64 dims)", font=dict(size=16)),
        xaxis=dict(title="Embedding Dimension", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        height=max(300, len(tokens) * 32),
    )
    return _styled(fig)


def plot_layer_norms(tokens: list[str], layer_outs: list[np.ndarray]) -> go.Figure:
    num_layers = len(layer_outs)
    norms = np.stack([np.linalg.norm(lo, axis=-1) for lo in layer_outs])

    fig = go.Figure(
        go.Heatmap(
            z=norms,
            x=tokens,
            y=[f"Layer {i}" for i in range(num_layers)],
            colorscale=_PURPLE,
            hovertemplate="Token: %{x}<br>%{y}<br>Norm: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Step 3 · Hidden-State Magnitude Across Layers", font=dict(size=16)),
        xaxis=dict(title="Tokens", tickangle=-45, showgrid=False),
        yaxis=dict(showgrid=False),
        height=max(300, num_layers * 30 + 100),
    )
    return _styled(fig)


def plot_attention_flow(tokens: list[str], attentions: list[np.ndarray]) -> go.Figure:
    num_layers = len(attentions)
    show_layers = min(num_layers, 4)
    indices = np.linspace(0, num_layers - 1, show_layers, dtype=int)

    fig = make_subplots(
        rows=1, cols=show_layers,
        subplot_titles=[f"Layer {i}" for i in indices],
        horizontal_spacing=0.04,
    )
    for col, li in enumerate(indices, 1):
        avg_attn = attentions[li].mean(axis=0)
        fig.add_trace(
            go.Heatmap(
                z=avg_attn, x=tokens, y=tokens,
                colorscale=_PURPLE, showscale=(col == show_layers),
                hovertemplate="From: %{y}<br>To: %{x}<br>Score: %{z:.3f}<extra></extra>",
            ),
            row=1, col=col,
        )
        fig.update_xaxes(tickangle=-45, showgrid=False, row=1, col=col)
        fig.update_yaxes(autorange="reversed", showgrid=False, row=1, col=col)

    fig.update_layout(
        title=dict(text="Step 4 · Attention Flow  (avg over heads)", font=dict(size=16)),
        height=max(400, len(tokens) * 35),
    )
    return _styled(fig)


def plot_final_output(tokens: list[str], logits_norm: np.ndarray) -> go.Figure:
    colors = [
        f"rgba(108, 99, 255, {0.3 + 0.7 * (v - logits_norm.min()) / (logits_norm.max() - logits_norm.min() + 1e-9)})"
        for v in logits_norm
    ]
    fig = go.Figure(
        go.Bar(
            x=tokens, y=logits_norm,
            marker_color=colors,
            hovertemplate="Token: %{x}<br>Norm: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Step 5 · Final Output Activation  (L2 norm per token)", font=dict(size=16)),
        xaxis=dict(title="Tokens", tickangle=-45, showgrid=False),
        yaxis=dict(title="Norm", showgrid=False),
        height=350,
    )
    return _styled(fig)
