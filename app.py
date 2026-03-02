"""
Attention Is All You Need — Interactive Transformer Explainer
=============================================================
Launch with:  python app.py
"""

import gradio as gr
import numpy as np

from utils.positional_encoding import generate_pe_matrix, plot_pe_heatmap
from utils.attention import get_attention_weights, plot_attention_heatmap
from utils.architecture import ENCODER_BLOCKS, DECODER_BLOCKS
from utils.pipeline_viz import (
    run_pipeline, plot_tokenization, plot_embedding_heatmap,
    plot_layer_norms, plot_attention_flow, plot_final_output,
)
from utils.rag_pipeline import ask as rag_ask


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.hero-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #A78BFA 40%, #F06292 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: 0.2rem; text-align: center;
}
.hero-subtitle {
    font-size: 1.15rem; color: #B0B0B0;
    text-align: center; margin-top: 0;
}
.section-header {
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(90deg, #6C63FF, #A78BFA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem; padding-bottom: 0.3rem;
    border-bottom: 2px solid rgba(108, 99, 255, 0.2);
}
.info-card {
    background: linear-gradient(145deg, #1A1A2E 0%, #16213E 100%);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 12px; padding: 1rem 1.2rem; margin: 0.6rem 0;
}
.equation-box {
    background: #1A1A2E; border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 12px; padding: 1rem 1.4rem; margin: 0.8rem 0;
    text-align: center; font-size: 1.1rem; color: #E0E0E0;
}
.arch-block {
    background: linear-gradient(145deg, #1A1A2E 0%, #16213E 100%);
    border-radius: 10px; padding: 0.8rem 1rem; margin: 0.4rem 0;
    border-left: 4px solid; transition: transform 0.2s;
}
.arch-block:hover { transform: translateX(4px); }
.arch-block h4 { margin: 0 0 0.3rem 0; color: #E0E0E0; }
.arch-block p { margin: 0; color: #B0B0B0; font-size: 0.9rem; }
.fancy-divider {
    height: 2px; border: none; margin: 1.5rem 0; border-radius: 2px;
    background: linear-gradient(90deg, transparent, #6C63FF, transparent);
}
footer { display: none !important; }
"""


# ═══════════════════════════════════════════════════════════════════════════
# CALLBACK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# -- Positional Encoding --
def update_pe(seq_len, d_model):
    pe = generate_pe_matrix(int(seq_len), int(d_model))
    return plot_pe_heatmap(pe, int(seq_len), int(d_model))


# -- Attention Visualiser --
_attn_cache = {}

def run_attention(sentence):
    if not sentence.strip():
        return None, gr.update(maximum=0), gr.update(maximum=0)
    tokens, attns = get_attention_weights(sentence)
    _attn_cache["tokens"] = tokens
    _attn_cache["attns"] = attns
    fig = plot_attention_heatmap(tokens, attns[0, 0], "Layer 0 · Head 0")
    n_layers = attns.shape[0]
    n_heads = attns.shape[1]
    return (
        fig,
        gr.update(maximum=n_layers - 1, value=0),
        gr.update(maximum=n_heads - 1, value=0),
    )


def update_head(layer_idx, head_idx):
    if "attns" not in _attn_cache:
        return None
    tokens = _attn_cache["tokens"]
    attns = _attn_cache["attns"]
    layer_idx = int(layer_idx)
    head_idx = int(head_idx)
    return plot_attention_heatmap(
        tokens, attns[layer_idx, head_idx],
        f"Layer {layer_idx} · Head {head_idx}",
    )


# -- Pipeline --
def run_full_pipeline(sentence):
    if not sentence.strip():
        empty = None
        return empty, empty, empty, empty, empty
    data = run_pipeline(sentence)
    t = data["tokens"]
    return (
        plot_tokenization(t, data["input_ids"]),
        plot_embedding_heatmap(t, data["embeddings"]),
        plot_layer_norms(t, data["layer_outs"]),
        plot_attention_flow(t, data["attentions"]),
        plot_final_output(t, data["logits_norm"]),
    )


# -- RAG Q&A --
def rag_respond(message, history):
    if not message.strip():
        return ""
    result = rag_ask(message)
    if result["error"]:
        return f"⚠️ {result['error']}"
    answer = result["answer"]
    if result["sources"]:
        chunks_html = ""
        for i, src in enumerate(result["sources"], 1):
            chunks_html += f"\n**Chunk {i}:**\n> {src}\n"
        answer += f"\n\n<details><summary>📄 Source Passages (click to expand)</summary>\n{chunks_html}\n</details>"
    return answer


# ═══════════════════════════════════════════════════════════════════════════
# BUILD THE GRADIO APP
# ═══════════════════════════════════════════════════════════════════════════

def build_arch_html(blocks, side_label):
    """Build HTML for one column of the architecture diagram."""
    html = f"<h3 style='text-align:center;color:#6C63FF;'>{side_label}</h3>"
    for b in blocks:
        color = b['color']
        name = b['name']
        short = b['short']
        detail = b['detail']
        code = b['code']
        html += (
            f"<details class='arch-block' style='border-left-color:{color};'>"
            f"<summary><b>{name}</b> — {short}</summary>"
            f"<div style='padding:0.6rem 0;'>"
            f"<p style='color:#C8C8C8;'>{detail}</p>"
            f"<pre style='background:#0E1117;padding:0.6rem;border-radius:8px;"
            f"overflow-x:auto;color:#A78BFA;font-size:0.85rem;'>{code}</pre>"
            f"</div></details>"
        )
    return html


with gr.Blocks(
    title="Transformer Explainer — Attention Is All You Need",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0E1117",
        body_text_color="#E0E0E0",
        block_background_fill="#16213E",
        block_border_color="rgba(108, 99, 255, 0.15)",
        block_label_text_color="#A78BFA",
        input_background_fill="#1A1A2E",
        button_primary_background_fill="linear-gradient(135deg, #6C63FF, #A78BFA)",
        button_primary_text_color="white",
    ),
) as demo:

    # ── Hero ──────────────────────────────────────────────────────────────
    gr.HTML("<p class='hero-title'>Attention Is All You Need</p>")
    gr.HTML(
        "<p class='hero-subtitle'>"
        "An interactive journey through the paper that started the LLM revolution"
        "</p>"
    )

    gr.HTML("<hr class='fancy-divider'>")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                "Published in **2017** by Vaswani *et al.*, this paper introduced the "
                "**Transformer** — a model built entirely on **attention mechanisms**, "
                "discarding the recurrent and convolutional layers that previously dominated "
                "sequence modelling.\n\n"
                "Every modern Large Language Model (GPT, LLaMA, Gemini, Claude, etc.) "
                "descends directly from this architecture. The core insight is elegantly "
                "simple: *let every word attend to every other word* and learn which "
                "relationships matter."
            )
        with gr.Column(scale=2):
            gr.HTML(
                '<div class="equation-box">'
                "<strong>The Central Equation</strong><br><br>"
                "Attention(Q, K, V) = softmax(QKᵀ / √d_k) V"
                "</div>"
            )

    gr.HTML("<hr class='fancy-divider'>")

    # ── TABS ──────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ┌────────────────────────────────────────────────────────────────┐
        # │ TAB 1: Positional Encoding                                      │
        # └────────────────────────────────────────────────────────────────┘
        with gr.TabItem("🌊 Positional Encoding"):
            gr.HTML("<p class='section-header'>Positional Encoding</p>")
            gr.Markdown(
                "Transformers process **all tokens simultaneously** — there is no built-in "
                "notion of word order. Positional encodings add sine/cosine waves to inject "
                "position information.\n\n"
                "**PE(pos, 2i) = sin(pos / 10000^(2i/d_model))**  ·  "
                "**PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))**"
            )

            with gr.Row():
                pe_seq = gr.Slider(10, 200, value=50, step=10, label="Sequence Length")
                pe_dim = gr.Slider(16, 512, value=128, step=16, label="Embedding Dimension (d_model)")

            pe_plot = gr.Plot(label="Positional Encoding Heatmap")
            pe_seq.change(update_pe, [pe_seq, pe_dim], pe_plot)
            pe_dim.change(update_pe, [pe_seq, pe_dim], pe_plot)
            demo.load(update_pe, [pe_seq, pe_dim], pe_plot)

            gr.HTML(
                '<div class="info-card">'
                "💡 <b>Key Insight:</b> Each position gets a unique &ldquo;fingerprint&rdquo; "
                "made of waves. Nearby positions have similar encodings, allowing the model "
                "to generalise to sequences longer than those seen during training."
                "</div>"
            )

        # ┌────────────────────────────────────────────────────────────────┐
        # │ TAB 2: Scaled Dot-Product Attention                              │
        # └────────────────────────────────────────────────────────────────┘
        with gr.TabItem("🔍 Attention"):
            gr.HTML("<p class='section-header'>Scaled Dot-Product Attention</p>")
            gr.Markdown(
                "For each token, the model asks: *\"Which other tokens should I pay attention to?\"*\n\n"
                "It computes **Query (Q)**, **Key (K)**, and **Value (V)** vectors, then:\n\n"
                "**Attention(Q, K, V) = softmax(QKᵀ / √d_k) V**\n\n"
                "Dividing by √d_k prevents the softmax from saturating in regions with vanishing gradients."
            )

            attn_input = gr.Textbox(
                label="🖊️ Type a sentence",
                value="The cat sat on the mat because it was tired",
                placeholder="Enter a sentence to analyse …",
            )
            attn_btn = gr.Button("Analyse Attention", variant="primary")

            attn_plot = gr.Plot(label="Attention Heatmap (Layer 0 · Head 0)")

            gr.HTML(
                '<div class="info-card">'
                "🔬 <b>Try this:</b> Hover over the word <code>it</code> on the Y-axis. "
                "Notice how the model assigns high attention to <code>cat</code> — "
                "it has learned to resolve the pronoun!"
                "</div>"
            )

            gr.HTML("<hr class='fancy-divider'>")

            # ── Multi-Head section (same tab) ──
            gr.HTML("<p class='section-header'>🧩 Multi-Head Attention</p>")
            gr.Markdown(
                "The Transformer runs **h parallel heads**, each learning to focus on "
                "*different types of relationships*:\n\n"
                "| Head | Might specialise in … |\n"
                "|------|----------------------|\n"
                "| Head 0 | Attending to the **next word** |\n"
                "| Head 1 | Subject ↔ Verb **agreement** |\n"
                "| Head 2 | **Preposition** ↔ Object links |\n"
                "| Head 3 | **Punctuation** & sentence boundaries |\n\n"
                "**MultiHead(Q,K,V) = Concat(head₁, …, headₕ) Wᴼ**"
            )

            with gr.Row():
                layer_slider = gr.Slider(0, 11, value=0, step=1, label="Layer")
                head_slider = gr.Slider(0, 11, value=0, step=1, label="Attention Head")

            mh_plot = gr.Plot(label="Multi-Head Attention Heatmap")

            attn_btn.click(
                run_attention, [attn_input],
                [attn_plot, layer_slider, head_slider],
            )
            layer_slider.change(update_head, [layer_slider, head_slider], mh_plot)
            head_slider.change(update_head, [layer_slider, head_slider], mh_plot)

            gr.HTML(
                '<div class="info-card">'
                "🧪 <b>Experiment:</b> Slide through the heads — you will see "
                "dramatically different patterns! Some heads focus on nearby tokens "
                "(local context), while others reach across the entire sentence."
                "</div>"
            )

        # ┌────────────────────────────────────────────────────────────────┐
        # │ TAB 3: Architecture Explorer                                    │
        # └────────────────────────────────────────────────────────────────┘
        with gr.TabItem("🏗️ Architecture"):
            gr.HTML("<p class='section-header'>Architecture Explorer</p>")
            gr.Markdown(
                "The Transformer follows an **Encoder–Decoder** structure. "
                "Click any block below to reveal its purpose, the underlying equation, "
                "and a PyTorch code snippet."
            )

            with gr.Row():
                with gr.Column():
                    gr.HTML(build_arch_html(ENCODER_BLOCKS, "Encoder ×N"))
                with gr.Column():
                    gr.HTML(build_arch_html(DECODER_BLOCKS, "Decoder ×N"))

        # ┌────────────────────────────────────────────────────────────────┐
        # │ TAB 4: Sentence Pipeline                                        │
        # └────────────────────────────────────────────────────────────────┘
        with gr.TabItem("⚡ Pipeline"):
            gr.HTML("<p class='section-header'>Sentence Pipeline</p>")
            gr.Markdown(
                "Type a sentence and follow it **step-by-step** through every stage "
                "of the Transformer encoder — from raw text to final contextual representations."
            )

            pipe_input = gr.Textbox(
                label="Enter a sentence",
                value="The cat sat on the mat because it was tired",
                placeholder="Type a sentence …",
            )
            pipe_btn = gr.Button("Run Full Pipeline", variant="primary")

            gr.Markdown("### 🔤 Step 1 — Tokenization")
            gr.Markdown("Raw sentence → sub-word tokens via WordPiece. `[CLS]` and `[SEP]` are added.")
            pipe_tok = gr.Plot(label="Tokenization")

            gr.Markdown("### 📐 Step 2 — Token + Positional Embeddings")
            gr.Markdown("Each token ID → 768-dim embedding vector, summed with its positional encoding.")
            pipe_emb = gr.Plot(label="Embeddings (first 64 dims)")

            gr.Markdown("### 📊 Step 3 — Hidden-State Magnitude Across Layers")
            gr.Markdown("L2 norm of each token's hidden state evolves as it passes through 12 encoder layers.")
            pipe_norms = gr.Plot(label="Layer Norms")

            gr.Markdown("### 🔀 Step 4 — Attention Flow")
            gr.Markdown("Average attention (across heads) at selected layers.")
            pipe_flow = gr.Plot(label="Attention Flow")

            gr.Markdown("### 🎯 Step 5 — Final Contextual Representation")
            gr.Markdown("Bar chart of the final hidden-state magnitude per token.")
            pipe_final = gr.Plot(label="Final Output")

            pipe_btn.click(
                run_full_pipeline, [pipe_input],
                [pipe_tok, pipe_emb, pipe_norms, pipe_flow, pipe_final],
            )

            gr.HTML(
                '<div class="info-card">'
                "🧠 <b>What you saw:</b> Your sentence went through "
                "<b>Tokenization → Embedding → 12 Self-Attention Layers → Final Output</b>. "
                "Each layer refined the representation by mixing information across tokens."
                "</div>"
            )

        # ┌────────────────────────────────────────────────────────────────┐
        # │ TAB 5: Paper Q&A (RAG)                                          │
        # └────────────────────────────────────────────────────────────────┘
        with gr.TabItem("💬 Paper Q&A"):
            gr.HTML("<p class='section-header'>Paper Q&A (RAG)</p>")
            gr.Markdown(
                "Ask any question about the *Attention Is All You Need* paper.\n\n"
                "> **Powered by:** `sentence-transformers/all-MiniLM-L6-v2` for embeddings "
                "and `google/flan-t5-base` for generation — **100% free, runs locally**.\n"
                "> The FAISS index is cached to disk after the first run."
            )

            qa_chatbot = gr.Chatbot(height=420, label="Conversation")
            with gr.Row():
                qa_input = gr.Textbox(
                    placeholder="Ask about the paper …",
                    label="Your question",
                    scale=5,
                    show_label=False,
                )
                qa_btn = gr.Button("Ask", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "Why divide by the square root of d_k?",
                    "What is multi-head attention?",
                    "How many parameters does the base model have?",
                    "What datasets were used for training?",
                ],
                inputs=qa_input,
            )

            def qa_submit(message, history):
                if not message.strip():
                    return history, ""
                history = history or []
                history.append({"role": "user", "content": message})
                bot_reply = rag_respond(message, history)
                history.append({"role": "assistant", "content": bot_reply})
                return history, ""

            qa_btn.click(qa_submit, [qa_input, qa_chatbot], [qa_chatbot, qa_input])
            qa_input.submit(qa_submit, [qa_input, qa_chatbot], [qa_chatbot, qa_input])

    # ── Footer ────────────────────────────────────────────────────────────
    gr.HTML("<hr class='fancy-divider'>")
    gr.HTML(
        "<p style='text-align:center;color:#666;font-size:0.85rem;'>"
        "Made with ❤️ using Gradio · "
        "Inspired by <em>Attention Is All You Need</em> (Vaswani et al., 2017)"
        "</p>"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch()
