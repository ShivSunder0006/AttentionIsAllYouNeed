"""Architecture Diagram — data & rendering for the clickable Transformer diagram."""

from __future__ import annotations
# ---------------------------------------------------------------------------

# Block definitions  (order = bottom → top of the encoder/decoder stack)
# ---------------------------------------------------------------------------

ENCODER_BLOCKS = [
    {
        "name": "Input Embedding",
        "color": "#6C63FF",
        "short": "Converts tokens to dense vectors.",
        "detail": (
            "Each token in the vocabulary is mapped to a learnable dense vector "
            "of size **d_model = 512**. These embeddings are shared between the "
            "encoder and decoder in the original paper."
        ),
        "code": (
            "import torch.nn as nn\n\n"
            "embedding = nn.Embedding(vocab_size, d_model)\n"
            "x = embedding(token_ids)  # (batch, seq_len, d_model)"
        ),
    },
    {
        "name": "Positional Encoding",
        "color": "#7C73FF",
        "short": "Injects word-order information via sine & cosine waves.",
        "detail": (
            "Since Transformers process all positions simultaneously, positional "
            "encoding adds a unique signal to each position using alternating "
            "**sin** and **cos** functions across embedding dimensions:\n\n"
            "$$PE_{(pos,2i)} = \\sin\\!\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right) "
            "\\qquad PE_{(pos,2i+1)} = \\cos\\!\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$"
        ),
        "code": (
            "import numpy as np\n\n"
            "angles = pos / np.power(10000, 2*(dims//2) / d_model)\n"
            "pe[:, 0::2] = np.sin(angles[:, 0::2])\n"
            "pe[:, 1::2] = np.cos(angles[:, 1::2])"
        ),
    },
    {
        "name": "Multi-Head Self-Attention",
        "color": "#F06292",
        "short": "Multiple parallel attention heads capture different relationships.",
        "detail": (
            "The input is linearly projected into **h** sets of queries, keys, and "
            "values. Each head independently computes scaled dot-product attention, "
            "then the outputs are concatenated and projected again:\n\n"
            "$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, \\dots, "
            "\\text{head}_h)\\,W^O$$\n\n"
            "where each $\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$."
        ),
        "code": (
            "import torch.nn as nn\n\n"
            "attn = nn.MultiheadAttention(d_model, num_heads)\n"
            "output, weights = attn(query, key, value)"
        ),
    },
    {
        "name": "Add & Norm",
        "color": "#4DB6AC",
        "short": "Residual connection + Layer Normalisation.",
        "detail": (
            "A **residual (skip) connection** adds the sub-layer's input back to "
            "its output, followed by **Layer Normalisation**.  This stabilises "
            "training and allows gradients to flow through many layers:\n\n"
            "$$\\text{output} = \\text{LayerNorm}(x + \\text{SubLayer}(x))$$"
        ),
        "code": (
            "import torch.nn as nn\n\n"
            "norm = nn.LayerNorm(d_model)\n"
            "output = norm(x + sublayer(x))"
        ),
    },
    {
        "name": "Feed-Forward Network",
        "color": "#FFB74D",
        "short": "Two linear layers with ReLU activation applied position-wise.",
        "detail": (
            "Each position's representation is passed through the same two-layer "
            "fully-connected network independently:\n\n"
            "$$\\text{FFN}(x) = \\text{ReLU}(xW_1 + b_1)\\,W_2 + b_2$$\n\n"
            "The inner dimension is typically **d_ff = 2048**, 4× the model dim."
        ),
        "code": (
            "import torch.nn as nn\n\n"
            "ffn = nn.Sequential(\n"
            "    nn.Linear(d_model, d_ff),\n"
            "    nn.ReLU(),\n"
            "    nn.Linear(d_ff, d_model),\n"
            ")"
        ),
    },
    {
        "name": "Add & Norm (2)",
        "color": "#4DB6AC",
        "short": "Second residual + LayerNorm after the FFN.",
        "detail": (
            "Exactly the same residual-connection + LayerNorm pattern applied "
            "after the Feed-Forward sub-layer:\n\n"
            "$$\\text{output} = \\text{LayerNorm}(x + \\text{FFN}(x))$$"
        ),
        "code": "output = norm2(x + ffn(x))",
    },
]

DECODER_BLOCKS = [
    {
        "name": "Output Embedding",
        "color": "#6C63FF",
        "short": "Same embedding layer as the encoder (weights are shared).",
        "detail": (
            "Target tokens are embedded identically to source tokens, sharing "
            "the same weight matrix."
        ),
        "code": "y = embedding(target_ids)",
    },
    {
        "name": "Positional Encoding",
        "color": "#7C73FF",
        "short": "Positional info added to the decoder input.",
        "detail": "Same positional-encoding scheme as the encoder side.",
        "code": "y = y + positional_encoding",
    },
    {
        "name": "Masked Multi-Head Attention",
        "color": "#EF5350",
        "short": "Self-attention with a causal mask to prevent looking ahead.",
        "detail": (
            "During training the decoder must not attend to future positions.  "
            "An upper-triangular mask sets those attention scores to **−∞** "
            "before the softmax, ensuring the model is autoregressive."
        ),
        "code": (
            "mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()\n"
            "attn_output, _ = attn(query, key, value, attn_mask=mask)"
        ),
    },
    {
        "name": "Add & Norm",
        "color": "#4DB6AC",
        "short": "Residual + LayerNorm after masked attention.",
        "detail": "$$\\text{output} = \\text{LayerNorm}(y + \\text{MaskedAttn}(y))$$",
        "code": "output = norm1(y + masked_attn(y))",
    },
    {
        "name": "Cross-Attention",
        "color": "#BA68C8",
        "short": "Attends to encoder output — bridges encoder and decoder.",
        "detail": (
            "Queries come from the decoder; keys and values come from the "
            "**encoder output**. This lets each decoder position attend over "
            "all positions in the input sequence."
        ),
        "code": "cross_out, _ = cross_attn(query=y, key=enc_out, value=enc_out)",
    },
    {
        "name": "Add & Norm",
        "color": "#4DB6AC",
        "short": "Residual + LayerNorm after cross-attention.",
        "detail": "$$\\text{output} = \\text{LayerNorm}(y + \\text{CrossAttn}(y, \\text{enc}))$$",
        "code": "output = norm2(y + cross_attn_out)",
    },
    {
        "name": "Feed-Forward Network",
        "color": "#FFB74D",
        "short": "Position-wise FFN (same structure as encoder).",
        "detail": "$$\\text{FFN}(x) = \\text{ReLU}(xW_1 + b_1)W_2 + b_2$$",
        "code": "output = ffn(output)",
    },
    {
        "name": "Add & Norm",
        "color": "#4DB6AC",
        "short": "Final residual + LayerNorm.",
        "detail": "$$\\text{output} = \\text{LayerNorm}(y + \\text{FFN}(y))$$",
        "code": "output = norm3(y + ffn_out)",
    },
    {
        "name": "Linear + Softmax",
        "color": "#42A5F5",
        "short": "Projects to vocabulary size and produces next-token probabilities.",
        "detail": (
            "A linear layer maps the decoder's final hidden state to "
            "**vocab_size** logits, then softmax produces a probability "
            "distribution over the vocabulary for the next token."
        ),
        "code": (
            "logits = nn.Linear(d_model, vocab_size)(output)\n"
            "probs  = torch.softmax(logits, dim=-1)"
        ),
    },
]


