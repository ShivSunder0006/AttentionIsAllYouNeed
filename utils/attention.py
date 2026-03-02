"""Attention utilities — load a pre-trained BERT model and extract attention weights."""

from __future__ import annotations

import functools
import torch
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Model loading (cached at module level)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_model(model_name: str = "bert-base-uncased"):
    """Return (tokenizer, model) with output_attentions enabled."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def get_attention_weights(sentence: str, model_name: str = "bert-base-uncased"):
    """
    Tokenize *sentence*, run a forward pass, and return:
        tokens  – list[str]  human-readable sub-word tokens
        attns   – Tensor of shape (num_layers, num_heads, seq, seq)
    """
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    attns = torch.stack(outputs.attentions).squeeze(1)  # (layers, heads, seq, seq)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, attns


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    tokens: list[str],
    weights_2d: torch.Tensor | None = None,
    title: str = "Attention Weights",
) -> go.Figure:
    """Plot a single (seq × seq) attention matrix as a Plotly heatmap."""
    if weights_2d is None:
        return go.Figure()

    z = weights_2d.cpu().numpy()

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=tokens,
            y=tokens,
            colorscale=[
                [0.0, "#0E1117"],
                [0.25, "#1A1A2E"],
                [0.5, "#6C63FF"],
                [0.75, "#A78BFA"],
                [1.0, "#E0CFFC"],
            ],
            colorbar=dict(title="Score", tickfont=dict(color="#E0E0E0")),
            hovertemplate="From: %{y}<br>To: %{x}<br>Score: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#E0E0E0")),
        xaxis=dict(
            title="Key Tokens", tickangle=-45,
            tickfont=dict(color="#B0B0B0", size=11),
            titlefont=dict(color="#E0E0E0"), showgrid=False, side="bottom",
        ),
        yaxis=dict(
            title="Query Tokens",
            tickfont=dict(color="#B0B0B0", size=11),
            titlefont=dict(color="#E0E0E0"), autorange="reversed", showgrid=False,
        ),
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        margin=dict(l=80, r=30, t=50, b=100),
        height=max(450, len(tokens) * 35),
        width=max(500, len(tokens) * 40),
    )
    return fig
