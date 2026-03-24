"""
app.py
------
Gradio UI for code_assistant_rag.

Tabs:
  1. Ingest Codebase  – upload a .zip of your project, auto-extract and ingest
  2. Ask the Codebase – chat with indexed codebase, see retrieved context + eval scores
"""

import os
import shutil
import sys
import tempfile
import threading
import zipfile
from io import StringIO

import gradio as gr
from dotenv import load_dotenv

from answer import answer_question
from ingest import ingest

load_dotenv(override=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def format_context(chunks) -> str:
    if not chunks:
        return "*No context retrieved.*"
    lines = ["## 📂 Retrieved Chunks\n"]
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        lines.append(f"---\n**`{source}`**\n\n{chunk.page_content}\n")
    return "\n".join(lines)


def format_eval(eval_result) -> str:
    if eval_result is None:
        return "*Evaluation will appear here after your first question.*"

    def bar(score: float) -> str:
        filled = round(score)
        return "🟩" * filled + "⬜" * (5 - filled)

    overall_color = (
        "🟢" if eval_result.overall >= 4.0
        else "🟡" if eval_result.overall >= 3.0
        else "🔴"
    )

    return f"""## {overall_color} Answer Quality

| Dimension | Score | |
|---|---|---|
| 🎯 Accuracy | {eval_result.accuracy:.1f} / 5 | {bar(eval_result.accuracy)} |
| 🔍 Relevance | {eval_result.relevance:.1f} / 5 | {bar(eval_result.relevance)} |
| 📋 Completeness | {eval_result.completeness:.1f} / 5 | {bar(eval_result.completeness)} |
| **Overall** | **{eval_result.overall:.1f} / 5** | {bar(eval_result.overall)} |

**Reasoning:** *{eval_result.reasoning}*
"""


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Ingest
# ══════════════════════════════════════════════════════════════════════════════

def run_ingest(zip_file, reset: bool):
    if zip_file is None:
        yield "⚠️ Please upload a .zip file of your project first."
        return

    tmp_dir = tempfile.mkdtemp(prefix="code_rag_")

    try:
        yield "📦 Extracting zip file...\n"

        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(tmp_dir)

        contents = os.listdir(tmp_dir)
        if len(contents) == 1 and os.path.isdir(os.path.join(tmp_dir, contents[0])):
            repo_path = os.path.join(tmp_dir, contents[0])
        else:
            repo_path = tmp_dir

        yield f"✅ Extracted successfully\n🔍 Scanning: `{repo_path}`...\n\n"

        output_buffer = StringIO()
        original_stdout = sys.stdout

        def run():
            sys.stdout = output_buffer
            try:
                ingest(repo_path, reset=reset)
            except Exception as e:
                output_buffer.write(f"\n❌ Error: {e}\n")
            finally:
                sys.stdout = original_stdout

        import time
        thread = threading.Thread(target=run)
        thread.start()

        while thread.is_alive():
            time.sleep(0.5)
            current = output_buffer.getvalue()
            if current:
                yield f"```\n{current}\n```"

        thread.join()
        final = output_buffer.getvalue()
        yield f"```\n{final}\n```\n\n✅ **Ingestion complete!** Switch to the **💬 Ask** tab to start querying."

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Chat
# ══════════════════════════════════════════════════════════════════════════════

def put_message_in_chatbot(message: str, history: list):
    return "", history + [{"role": "user", "content": message}]


def chat(history: list):
    if not history:
        return history, "*No context yet.*", "*No evaluation yet.*"

    last_message = history[-1]["content"]
    prior = history[:-1]

    answer, chunks, eval_result = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})

    return history, format_context(chunks), format_eval(eval_result)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

DARK_BG      = "#0f1117"
BLOCK_BG     = "#1a1d27"
INPUT_BG     = "#12151f"
BORDER       = "#2d3148"
TEXT         = "#e2e8f0"
TEXT_SUB     = "#94a3b8"
TEXT_MUTED   = "#64748b"
ACCENT       = "#6366f1"
ACCENT_HOVER = "#4f46e5"
CODE_GREEN   = "#a5f3c0"

# JS injected on load — MutationObserver forces dark styles on Gradio's
# hardcoded-white file upload table which cannot be overridden by CSS alone.
DARK_PATCH_JS = """
() => {
    function patchDark() {
        const selectors = [
            '.file-preview-holder',
            '.file-preview-holder table',
            '.file-preview-holder tr',
            '.file-preview-holder td',
            '.file-preview-holder th',
            '.file-component',
            '.upload-container',
        ];
        selectors.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => {
                el.style.setProperty('background-color', '#1a1d27', 'important');
                el.style.setProperty('color', '#e2e8f0', 'important');
                el.style.setProperty('border-color', '#2d3148', 'important');
            });
        });
    }
    patchDark();
    const observer = new MutationObserver(patchDark);
    observer.observe(document.body, { childList: true, subtree: true });
}
"""

def main():
    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill=DARK_BG,
        body_background_fill_dark=DARK_BG,
        block_background_fill=BLOCK_BG,
        block_background_fill_dark=BLOCK_BG,
        block_border_color=BORDER,
        block_border_color_dark=BORDER,
        block_label_text_color=TEXT_SUB,
        block_label_text_color_dark=TEXT_SUB,
        input_background_fill=INPUT_BG,
        input_background_fill_dark=INPUT_BG,
        input_border_color=BORDER,
        input_border_color_dark=BORDER,
        button_primary_background_fill=ACCENT,
        button_primary_background_fill_dark=ACCENT,
        button_primary_background_fill_hover=ACCENT_HOVER,
        button_primary_background_fill_hover_dark=ACCENT_HOVER,
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        body_text_color=TEXT,
        body_text_color_dark=TEXT,
        body_text_color_subdued=TEXT_SUB,
        body_text_color_subdued_dark=TEXT_SUB,
    )

    css = f"""
    /* ── Global ── */
    .gradio-container {{ max-width: 1400px !important; }}
    footer {{ display: none !important; }}
    * {{ box-sizing: border-box; }}

    /* ── Tabs ── */
    .tab-nav button {{
        font-size: 1rem !important;
        padding: 10px 20px !important;
        color: {TEXT_SUB} !important;
        background: transparent !important;
    }}
    .tab-nav button.selected {{
        border-bottom: 2px solid {ACCENT} !important;
        color: {ACCENT} !important;
    }}

    /* ── All blocks and containers ── */
    .block, .block.padded, .block.border,
    .container, .wrap, .wrap.default,
    .prose, .md, .markdown,
    div[data-testid="markdown"],
    div[data-testid="chatbot"],
    .chatbot, .chatbot > div,
    .overflow-y-auto, .gap, .form {{
        background-color: {BLOCK_BG} !important;
        color: {TEXT} !important;
    }}

    /* ── File upload ── */
    .file-preview-holder,
    .file-preview-holder *,
    div[data-testid="file"],
    div[data-testid="file"] *,
    div[data-testid="file-upload"],
    div[data-testid="file-upload"] *,
    .file-component, .file-component *,
    table, table tr, table td, table th {{
        background-color: {BLOCK_BG} !important;
        color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}

    /* ── Code blocks ── */
    pre, code, .code {{
        background-color: {INPUT_BG} !important;
        color: {CODE_GREEN} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
    }}

    /* ── Markdown tables ── */
    .prose table th {{ background-color: {INPUT_BG} !important; }}
    .prose table tr:nth-child(even) td {{ background-color: #161923 !important; }}

    /* ── Chatbot messages ── */
    .message-wrap, .message-wrap > div {{ background-color: {BLOCK_BG} !important; }}
    .message.user {{ background-color: #1e2235 !important; color: {TEXT} !important; border-radius: 8px !important; }}
    .message.bot  {{ background-color: {INPUT_BG} !important; color: {TEXT} !important; border-radius: 8px !important; }}

    /* ── Inputs ── */
    textarea, input[type="text"] {{
        background-color: {INPUT_BG} !important;
        color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}
    textarea::placeholder, input::placeholder {{ color: #4a5568 !important; }}

    /* ── Checkbox & labels ── */
    input[type="checkbox"] {{ accent-color: {ACCENT}; }}
    label {{ color: {TEXT} !important; }}
    .form span {{ color: {TEXT_SUB} !important; }}

    /* ── Scrollbars ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BLOCK_BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}
    """

    with gr.Blocks(title="AI Code Assistant", theme=theme, css=css) as ui:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div style="padding: 28px 0 18px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
                <span style="font-size: 2rem;">🤖</span>
                <h1 style="font-size: 2rem; font-weight: 700; color: {TEXT}; margin: 0;">
                    AI Code Assistant
                </h1>
            </div>
            <p style="color: {TEXT_SUB}; font-size: 1rem; margin: 0;">
                Ingest any codebase. Ask anything about it. &nbsp;·&nbsp;
                <span style="color: {ACCENT};">GPT-4.1</span> &nbsp;·&nbsp;
                <span style="color: {ACCENT};">ChromaDB</span> &nbsp;·&nbsp;
                <span style="color: {ACCENT};">text-embedding-3-large</span>
            </p>
        </div>
        """)

        with gr.Tabs():

            # ── TAB 1: INGEST ─────────────────────────────────────────────────
            with gr.Tab("📁  Ingest Codebase"):

                gr.HTML(f"""
                <div style="padding: 20px 0 8px 0;">
                    <h2 style="font-size: 1.4rem; font-weight: 600; color: {TEXT}; margin: 0 0 8px 0;">
                        1. Add your project
                    </h2>
                    <p style="color: {TEXT_SUB}; margin: 0 0 6px 0;">
                        Upload your project as a
                        <code style="color:{ACCENT}; background:{INPUT_BG}; padding: 2px 6px; border-radius:4px;">.zip</code>
                        file. The pipeline will scan every source file, split it into semantic chunks using an LLM,
                        generate embeddings, and store everything in a local ChromaDB vector database.
                    </p>
                    <p style="color: {TEXT_MUTED}; font-size: 0.875rem; margin: 0;">
                        <strong style="color:{TEXT_SUB};">Supported:</strong>
                        .py .js .ts .jsx .tsx .md .json .yaml .html .css .sh .sql and more &nbsp;·&nbsp;
                        <strong style="color:{TEXT_SUB};">Auto-ignored:</strong>
                        .git &nbsp; node_modules &nbsp; .venv &nbsp; __pycache__ &nbsp; build &nbsp; dist
                    </p>
                </div>
                """)

                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        zip_input = gr.File(
                            label="📦 Upload Project ZIP",
                            file_types=[".zip"],
                            type="filepath",
                            height=160,
                        )
                    with gr.Column(scale=1):
                        gr.HTML(f"""
                        <div style="padding: 16px 0 8px 0;">
                            <h3 style="font-size: 1rem; color: {TEXT}; margin: 0 0 12px 0;">2. Options</h3>
                        </div>
                        """)
                        reset_checkbox = gr.Checkbox(
                            label="🗑️ Reset existing index",
                            value=False,
                            info="Wipe ChromaDB before ingesting (use when re-indexing a project)",
                        )

                ingest_button = gr.Button(
                    "🚀  Start Ingestion", variant="primary", size="lg"
                )

                gr.HTML(f"""
                <div style="padding: 16px 0 4px 0;">
                    <h3 style="font-size: 1rem; color: {TEXT}; margin: 0 0 4px 0;">3. Ingestion Log</h3>
                </div>
                """)

                ingest_output = gr.Markdown(
                    value="*Upload a `.zip` and click **Start Ingestion** to begin.*",
                    container=True,
                    height=300,
                )

                ingest_button.click(
                    fn=run_ingest,
                    inputs=[zip_input, reset_checkbox],
                    outputs=ingest_output,
                )

            # ── TAB 2: ASK ────────────────────────────────────────────────────
            with gr.Tab("💬  Ask the Codebase"):

                gr.HTML(f"""
                <div style="padding: 20px 0 12px 0;">
                    <h2 style="font-size: 1.4rem; font-weight: 600; color: {TEXT}; margin: 0 0 6px 0;">
                        Ask anything about your indexed project
                    </h2>
                    <p style="color: {TEXT_MUTED}; font-size: 0.9rem; margin: 0;">
                        Try: &nbsp;
                        <em style="color:{TEXT_SUB};">"Where is the function that creates embeddings?"</em> &nbsp;·&nbsp;
                        <em style="color:{TEXT_SUB};">"How does the retry logic work?"</em> &nbsp;·&nbsp;
                        <em style="color:{TEXT_SUB};">"What files handle database connections?"</em>
                    </p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="💬 Conversation",
                            height=500,
                            type="messages",
                            show_copy_button=True,
                            placeholder="Ask about any function, module, or concept in your codebase.",
                        )
                        with gr.Row():
                            message_input = gr.Textbox(
                                placeholder="e.g. Where is the rerank function and how does it work?",
                                show_label=False,
                                lines=2,
                                scale=4,
                            )
                            send_btn = gr.Button("Send ↩", variant="primary", scale=1)

                    with gr.Column(scale=1):
                        eval_panel = gr.Markdown(
                            value="*Answer quality scores will appear here.*",
                            label="📊 Answer Quality",
                            container=True,
                            height=220,
                        )
                        context_panel = gr.Markdown(
                            value="*Retrieved code chunks will appear here.*",
                            label="📂 Retrieved Context",
                            container=True,
                            height=310,
                        )

                message_input.submit(
                    fn=put_message_in_chatbot,
                    inputs=[message_input, chatbot],
                    outputs=[message_input, chatbot],
                ).then(
                    fn=chat,
                    inputs=chatbot,
                    outputs=[chatbot, context_panel, eval_panel],
                )

                send_btn.click(
                    fn=put_message_in_chatbot,
                    inputs=[message_input, chatbot],
                    outputs=[message_input, chatbot],
                ).then(
                    fn=chat,
                    inputs=chatbot,
                    outputs=[chatbot, context_panel, eval_panel],
                )

        # ── Inject JS dark patch on load ──────────────────────────────────────
        ui.load(fn=None, js=DARK_PATCH_JS)

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()