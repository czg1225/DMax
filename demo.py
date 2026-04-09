import argparse
import html
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "Zigeng/DMax-Math-16B"
DEFAULT_PROMPT = "Solve 37 * 48 and explain the intermediate reasoning briefly."

STAGE_SIDEBAR_WIDTH = 320
STAGE_WIDTH = 720
STAGE_MIN_HEIGHT = 620
STAGE_FINAL_HEIGHT_VH = 72
STAGE_FINAL_MAX_HEIGHT = 860
TOKEN_GRID_MIN_HEIGHT = 600
FINAL_MARKDOWN_BASE_FONT_SIZE = 20
FINAL_MARKDOWN_MIN_FONT_SIZE = 11
PLAYBACK_INTERVAL_MS = 180


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the full step-by-step decoding process of the diffusion LM."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--gsm8k-index", type=int, default=0)
    parser.add_argument("--gen-length", type=int, default=512)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output", default="dllm_demo.html")
    return parser.parse_args()


def load_prompt(args):
    if args.prompt is not None:
        return args.prompt

    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        return ds[args.gsm8k_index]["question"] + "\nLet's think step by step\n"
    except Exception:
        return DEFAULT_PROMPT


def tokenize_prompt(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    return tokenizer(prompt, return_tensors="pt").input_ids


def format_token_label(tokenizer, token_id, mask_id, eos_id):
    if token_id == mask_id:
        return "[MASK]"
    if eos_id is not None and token_id == eos_id:
        return "[EOS]"

    text = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not text:
        text = tokenizer.convert_ids_to_tokens(token_id)
    if text is None:
        text = str(token_id)

    text = (
        text.replace("\n", "↵")
        .replace("\t", "⇥")
        .replace("\r", "␍")
        .replace(" ", "·")
    )
    if text == "":
        text = "∅"
    if len(text) > 20:
        text = text[:17] + "..."
    return text


def collect_token_labels(tokenizer, demo_trace):
    used_ids = set(demo_trace["prompt_token_ids"])
    used_ids.update(demo_trace["generated_token_ids"])
    used_ids.update(demo_trace["final_token_ids"])
    for frame in demo_trace["frames"]:
        used_ids.update(frame.get("visible_generated_ids", []))
        used_ids.update(frame.get("pre_visible_ids", []))
        used_ids.update(frame.get("post_visible_ids", []))
        used_ids.update(frame.get("top1_token_ids", []))
        used_ids.update(frame.get("block_top1_token_ids", []))

    mask_id = demo_trace["mask_id"]
    eos_id = demo_trace["eos_id"]
    token_labels = {}
    for token_id in sorted(used_ids):
        token_labels[str(token_id)] = format_token_label(
            tokenizer, token_id, mask_id, eos_id
        )
    return token_labels


def enrich_demo_trace_for_render(demo_trace):
    prompt_length = demo_trace["prompt_length"]
    original_frames = demo_trace["frames"]
    if not original_frames:
        return demo_trace

    enriched_frames = []

    for frame in original_frames:
        generated_before_ids = frame["pre_visible_ids"][prompt_length:]
        generated_after_ids = frame["post_visible_ids"][prompt_length:]

        current_block_abs_start = max(prompt_length, frame["block_start"])
        current_block_generated_start = max(0, current_block_abs_start - prompt_length)
        prompt_overlap = max(0, prompt_length - frame["block_start"])
        visible_generated_absolute_positions = list(
            range(prompt_length, frame["window_end"])
        )

        frame["generated_before_ids"] = generated_before_ids
        frame["generated_after_ids"] = generated_after_ids
        frame["visible_generated_ids"] = generated_after_ids
        frame["visible_generated_absolute_positions"] = visible_generated_absolute_positions
        frame["current_block_absolute_start"] = current_block_abs_start
        frame["current_block_generated_start"] = current_block_generated_start
        frame["block_top1_confidence"] = frame["top1_confidence"][prompt_overlap:]
        frame["block_top1_token_ids"] = frame["top1_token_ids"][prompt_overlap:]
        frame["block_input_confidence"] = frame["input_confidence"][prompt_overlap:]
        frame["block_mask_index_before"] = frame["mask_index_before"][prompt_overlap:]
        frame["block_token_index_before"] = frame["token_index_before"][prompt_overlap:]
        frame["block_active_mask"] = frame["active_block_mask"][prompt_overlap:]
        frame["block_absolute_positions"] = list(
            range(current_block_abs_start, frame["block_end"])
        )
        frame["block_prompt_overlap"] = prompt_overlap
        frame["block_decoded_positions"] = [
            pos - prompt_overlap
            for pos in frame["decoded_positions"]
            if pos >= prompt_overlap
        ]
        frame["decoded_absolute_positions"] = [
            current_block_abs_start + pos for pos in frame["block_decoded_positions"]
        ]
        frame["state_mode"] = "after_step"
        enriched_frames.append(frame)

    first_frame = enriched_frames[0]
    initial_frame = {
        "frame_id": -1,
        "block_id": first_frame["block_id"],
        "absolute_block_id": first_frame["absolute_block_id"],
        "step_id": -1,
        "window_end": first_frame["window_end"],
        "block_start": first_frame["block_start"],
        "block_end": first_frame["block_end"],
        "nfe": 0,
        "visible_generated_ids": first_frame["generated_before_ids"],
        "visible_generated_absolute_positions": list(
            range(prompt_length, first_frame["window_end"])
        ),
        "current_block_absolute_start": first_frame["current_block_absolute_start"],
        "current_block_generated_start": first_frame["current_block_generated_start"],
        "block_top1_confidence": [],
        "block_top1_token_ids": [],
        "block_input_confidence": first_frame["block_input_confidence"],
        "block_mask_index_before": first_frame["block_mask_index_before"],
        "block_token_index_before": first_frame["block_token_index_before"],
        "block_active_mask": first_frame["block_active_mask"],
        "block_absolute_positions": first_frame["block_absolute_positions"],
        "block_prompt_overlap": first_frame["block_prompt_overlap"],
        "block_decoded_positions": [],
        "decoded_absolute_positions": [],
        "same_as_previous": False,
        "all_confident": False,
        "converged": False,
        "convergence_reason": None,
        "state_mode": "before_first_step",
    }

    demo_trace["frames"] = [initial_frame]
    for idx, frame in enumerate(enriched_frames, start=1):
        frame["frame_id"] = idx
        demo_trace["frames"].append(frame)

    return demo_trace


def markdown_to_html(text):
    text = re.sub(r"\n{2,}", "\n", text)
    escaped = html.escape(text.replace("\r\n", "\n"))
    code_blocks = []

    def stash_code(match):
        code = match.group(1).strip("\n")
        placeholder = f"@@CODEBLOCK{len(code_blocks)}@@"
        code_blocks.append(f"<pre><code>{code}</code></pre>")
        return placeholder

    escaped = re.sub(r"```(?:[^\n`]*)\n(.*?)```", stash_code, escaped, flags=re.S)

    def format_inline(content):
        content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
        content = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", content)
        content = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", content)
        return content

    parts = []
    paragraph_lines = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            parts.append("</ul>")
            in_ul = False
        if in_ol:
            parts.append("</ol>")
            in_ol = False

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            merged = " ".join(line.strip() for line in paragraph_lines if line.strip())
            if merged:
                parts.append(f"<p>{format_inline(merged)}</p>")
        paragraph_lines = []

    for raw_line in escaped.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            close_lists()
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        unordered = re.match(r"^[-*+]\s+(.*)$", stripped)
        ordered = re.match(r"^(\d+)\.\s+(.*)$", stripped)

        if heading:
            flush_paragraph()
            close_lists()
            level = len(heading.group(1))
            parts.append(f"<h{level}>{format_inline(heading.group(2))}</h{level}>")
        elif unordered:
            flush_paragraph()
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            if not in_ul:
                parts.append("<ul>")
                in_ul = True
            parts.append(f"<li>{format_inline(unordered.group(1))}</li>")
        elif ordered:
            flush_paragraph()
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if not in_ol:
                parts.append(f'<ol start="{ordered.group(1)}">')
                in_ol = True
            parts.append(f"<li>{format_inline(ordered.group(2))}</li>")
        else:
            close_lists()
            paragraph_lines.append(stripped)

    flush_paragraph()
    close_lists()
    rendered = "\n".join(parts)
    for idx, block in enumerate(code_blocks):
        rendered = rendered.replace(f"@@CODEBLOCK{idx}@@", block)
    return rendered


def build_demo_payload(prompt, generated_answer, demo_trace, nfe, token_labels):
    generated_token_count = len(demo_trace["generated_token_ids"])
    tpf = generated_token_count / nfe if nfe else 0.0
    return {
        "title": "Diffusion Decoding Demo",
        "prompt": prompt,
        "generated_answer": generated_answer,
        "nfe": nfe,
        "tpf": tpf,
        "prompt_length": demo_trace["prompt_length"],
        "block_length": demo_trace["block_length"],
        "steps": demo_trace["steps"],
        "threshold": demo_trace["threshold"],
        "frames": demo_trace["frames"],
        "blocks": demo_trace["blocks"],
        "token_labels": token_labels,
        "mask_id": demo_trace["mask_id"],
        "eos_id": demo_trace["eos_id"],
        "generated_token_ids": demo_trace["generated_token_ids"],
        "generated_html": markdown_to_html(generated_answer),
    }


def render_html(payload):
    payload_json = json.dumps(payload, ensure_ascii=False)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Diffusion Decoding Demo</title>
  <style>
    :root {{
      --bg: #f5efe5;
      --bg-soft: #f9f5ee;
      --panel: rgba(255, 252, 247, 0.82);
      --ink: #1d1b18;
      --muted: #6d665d;
      --line: rgba(68, 55, 39, 0.12);
      --accent: #b85c38;
      --accent-2: #0f766e;
      --accent-3: #6d4aa2;
      --prompt: #d9e6f2;
      --active-token: #d7efe7;
      --active-mask: #f8e3b3;
      --decoded: #ffd7c2;
      --stable: #dde8ff;
      --shadow: 0 24px 60px rgba(102, 73, 31, 0.12);
      --radius: 22px;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(184, 92, 56, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.10), transparent 24%),
        linear-gradient(180deg, #fcf8f2 0%, var(--bg) 100%);
    }}

    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}

    .hero {{
      display: grid;
      grid-template-columns: 1.35fr 0.95fr;
      gap: 20px;
      margin-bottom: 20px;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}

    .hero-main {{
      padding: 24px 26px;
      position: relative;
      overflow: hidden;
    }}

    .hero-main::after {{
      content: "";
      position: absolute;
      inset: auto -10% -45% auto;
      width: 280px;
      height: 280px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(184, 92, 56, 0.12), transparent 68%);
      pointer-events: none;
    }}

    .eyebrow {{
      margin: 0 0 8px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 12px;
      color: var(--accent);
      font-weight: 700;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(32px, 5vw, 56px);
      line-height: 0.95;
      letter-spacing: -0.03em;
    }}

    .hero-copy {{
      margin-top: 14px;
      max-width: 60ch;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
    }}

    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      padding: 20px;
    }}

    .metric {{
      padding: 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.64);
      border: 1px solid var(--line);
    }}

    .metric-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}

    .metric-value {{
      font-size: 28px;
      line-height: 1;
      font-weight: 700;
    }}

    .metric-sub {{
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
    }}

    .content {{
      display: grid;
      grid-template-columns: __STAGE_SIDEBAR_WIDTH__px __STAGE_WIDTH__px;
      justify-content: center;
      gap: 20px;
    }}

    .sidebar,
    .workspace {{
      padding: 20px;
    }}

    .section-title {{
      margin: 0 0 12px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
    }}

    .prompt-box,
    .answer-box {{
      padding: 16px 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.7);
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
      color: #2b2621;
    }}

    .answer-box {{
      width: 100%;
      height: 100%;
      min-height: 0;
      font-size: __FINAL_MARKDOWN_BASE_FONT_SIZE__px;
      line-height: 1.08;
      padding: 10px 12px;
      overflow: hidden;
    }}

    .answer-box h1,
    .answer-box h2,
    .answer-box h3,
    .answer-box h4 {{
      margin: 0;
      line-height: 1.02;
      font-size: 1em;
      font-weight: 700;
    }}

    .answer-box p,
    .answer-box ul,
    .answer-box ol,
    .answer-box pre {{
      margin: 0;
    }}

    .answer-box ul,
    .answer-box ol {{
      padding-left: 0.95em;
    }}

    .answer-box li {{
      margin: 0;
    }}

    .answer-box code {{
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      background: rgba(31, 92, 87, 0.08);
      padding: 0.03em 0.16em;
      border-radius: 4px;
    }}

    .answer-box pre {{
      overflow: hidden;
      padding: 4px 6px;
      border-radius: 6px;
      background: rgba(29, 27, 24, 0.06);
    }}

    .answer-box pre code {{
      background: transparent;
      padding: 0;
    }}

    .answer-box > * + * {{
      margin-top: 0 !important;
    }}

    .answer-box br {{
      display: none;
    }}

    .controls {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}

    button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: #1f5c57;
      color: white;
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}

    button.secondary {{
      background: rgba(31, 92, 87, 0.12);
      color: #1f5c57;
    }}

    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      background: rgba(184, 92, 56, 0.12);
      color: var(--accent);
    }}

    .token-section {{
      margin-top: 14px;
      width: 100%;
    }}

    .token-section h3 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}

    .stage-box {{
      width: 100%;
      min-height: __STAGE_MIN_HEIGHT__px;
      max-width: __STAGE_WIDTH__px;
      margin: 0 auto;
      padding: 10px;
      border-radius: 22px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.6);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.45);
      overflow: visible;
      display: block;
    }}

    .stage-box.final-stage {{
      height: __STAGE_FINAL_HEIGHT_VH__vh;
      max-height: __STAGE_FINAL_MAX_HEIGHT__px;
      overflow: hidden;
    }}

    .token-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: flex-start;
      align-content: flex-start;
      width: 100%;
      height: auto;
      min-height: __TOKEN_GRID_MIN_HEIGHT__px;
      overflow: visible;
    }}

    .token {{
      position: relative;
      min-width: 38px;
      max-width: 96px;
      padding: 6px 7px 5px;
      border-radius: 12px;
      border: 1px solid rgba(53, 44, 33, 0.12);
      background: rgba(255, 255, 255, 0.88);
      overflow: hidden;
      box-shadow: 0 6px 14px rgba(61, 48, 31, 0.05);
    }}

    .token::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: calc(var(--conf, 0) * 100%);
      background: linear-gradient(90deg, rgba(184, 92, 56, 0.14), rgba(15, 118, 110, 0.20));
      pointer-events: none;
    }}

    .token.prompt {{ background: var(--prompt); }}
    .token.mask {{ background: var(--active-mask); }}
    .token.token-index {{ background: var(--stable); }}
    .token.decoded {{ background: var(--decoded); }}
    .token.refreshed {{ background: var(--active-token); }}
    .token.committed {{ background: rgba(255, 255, 255, 0.96); }}
    .token.current-block {{
      outline: 2px solid rgba(109, 74, 162, 0.18);
      outline-offset: 1px;
    }}

    .token-label {{
      position: relative;
      z-index: 1;
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 10px;
      line-height: 1.15;
      word-break: break-word;
      white-space: pre-wrap;
    }}

    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}

    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }}

    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(0, 0, 0, 0.08);
    }}

    .token-caption {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }}

    .legend[hidden],
    .token-caption[hidden],
    .token-grid[hidden],
    .answer-box[hidden],
    .stage-placeholder[hidden] {{
      display: none !important;
    }}

    .stage-placeholder {{
      width: 100%;
      height: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px dashed var(--line);
      color: var(--muted);
      background: rgba(255, 255, 255, 0.58);
      font-size: 13px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
    }}

    @media (max-width: 1100px) {{
      .hero,
      .content {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="panel hero-main">
        <p class="eyebrow">Diffusion Language Model</p>
        <h1>Step-by-step decoding,<br />made visible.</h1>
        <div class="hero-copy">
          Inspect the full block-wise diffusion process frame by frame, including token refreshes,
          left-to-right mask decoding, confidence evolution, and block convergence.
        </div>
      </div>
      <div class="panel summary-grid">
        <div class="metric">
          <div class="metric-label">TPF</div>
          <div class="metric-value" id="metric-frames"></div>
          <div class="metric-sub">Generated tokens / NFE</div>
        </div>
        <div class="metric">
          <div class="metric-label">NFE</div>
          <div class="metric-value" id="metric-nfe"></div>
          <div class="metric-sub">Forward passes</div>
        </div>
        <div class="metric">
          <div class="metric-label">Generated Tokens</div>
          <div class="metric-value" id="metric-gen"></div>
          <div class="metric-sub">Final visible answer length</div>
        </div>
        <div class="metric">
          <div class="metric-label">Threshold</div>
          <div class="metric-value" id="metric-threshold"></div>
          <div class="metric-sub">Mask decode cutoff</div>
        </div>
      </div>
    </section>

    <section class="content">
      <aside class="panel sidebar">
        <h2 class="section-title">Prompt</h2>
        <div class="prompt-box" id="prompt-box"></div>
      </aside>

      <main class="panel workspace">
        <div class="controls">
          <button id="play-btn">Play</button>
          <button id="prev-btn" class="secondary">Prev</button>
          <button id="next-btn" class="secondary">Next</button>
          <span class="badge" id="convergence-badge">Running</span>
        </div>
        <input id="frame-slider" type="range" min="0" max="0" value="0" />

        <section class="token-section">
          <h3 id="stage-title">Visible generated tokens so far</h3>
          <div class="stage-box">
            <div class="token-grid" id="tokens-step"></div>
            <div class="answer-box" id="answer-box" hidden></div>
            <div class="stage-placeholder" id="stage-placeholder" hidden></div>
          </div>
          <p class="token-caption" id="token-caption">Previously decoded blocks stay visible and the raw token sequence grows over time. The final generated answer is rendered as markdown only after the full sequence finishes decoding.</p>
        </section>

        <div class="legend">
          <span><i class="swatch" style="background: var(--active-mask);"></i>Mask before step</span>
          <span><i class="swatch" style="background: var(--stable);"></i>Existing token index</span>
          <span><i class="swatch" style="background: var(--active-token);"></i>Refreshed token</span>
          <span><i class="swatch" style="background: var(--decoded);"></i>Decoded mask</span>
          <span><i class="swatch" style="background: rgba(255, 255, 255, 0.96);"></i>Committed previous block</span>
        </div>
      </main>
    </section>
  </div>

  <script>
    const DATA = __PAYLOAD_JSON__;
    const slider = document.getElementById("frame-slider");
    const playBtn = document.getElementById("play-btn");
    const prevBtn = document.getElementById("prev-btn");
    const nextBtn = document.getElementById("next-btn");
    const promptBox = document.getElementById("prompt-box");
    const answerBox = document.getElementById("answer-box");
    const metricFrames = document.getElementById("metric-frames");
    const metricNfe = document.getElementById("metric-nfe");
    const metricGen = document.getElementById("metric-gen");
    const metricThreshold = document.getElementById("metric-threshold");
    const convergenceBadge = document.getElementById("convergence-badge");
    const stageTitle = document.getElementById("stage-title");
    const stageBox = document.querySelector(".stage-box");
    const tokensStep = document.getElementById("tokens-step");
    const stagePlaceholder = document.getElementById("stage-placeholder");
    const tokenCaption = document.getElementById("token-caption");
    const legend = document.querySelector(".legend");

    let currentFrame = 0;
    let timer = null;

    function tokenLabel(tokenId) {{
      return DATA.token_labels[String(tokenId)] ?? String(tokenId);
    }}

    function formatReason(reason) {{
      if (reason === "stable_tokens") return "Stable tokens";
      if (reason === "high_confidence") return "High confidence";
      return "Running";
    }}

    function buildTokenEl(tokenId, localIndex, frame) {{
      const el = document.createElement("div");
      const decodedSet = new Set(frame.decoded_absolute_positions || []);
      const absolutePos = frame.visible_generated_absolute_positions[localIndex];
      const inCurrentBlock =
        absolutePos >= frame.current_block_absolute_start && absolutePos < frame.block_end;
      const blockLocalIndex = inCurrentBlock ? absolutePos - frame.current_block_absolute_start : -1;
      const inputConf =
        inCurrentBlock && blockLocalIndex < frame.block_input_confidence.length
          ? Number(frame.block_input_confidence[blockLocalIndex] || 0)
          : 0;
      const top1Conf =
        inCurrentBlock && blockLocalIndex < frame.block_top1_confidence.length
          ? Number(frame.block_top1_confidence[blockLocalIndex] || 0)
          : 0;

      let cls = [];
      if (inCurrentBlock) {{
        cls.push("current-block");
        if (tokenId === DATA.mask_id) cls.push("mask");
        if (decodedSet.has(absolutePos)) cls.push("decoded");
        else if (
          blockLocalIndex >= 0 &&
          frame.block_token_index_before[blockLocalIndex]
        ) cls.push("refreshed");
        else if (
          blockLocalIndex >= 0 &&
          frame.block_mask_index_before[blockLocalIndex]
        ) cls.push("mask");
        else cls.push("token-index");
      }} else {{
        cls.push("committed");
      }}

      el.className = `token ${cls.join(" ")}`.trim();
      el.style.setProperty("--conf", top1Conf);
      el.title = `abs pos: ${absolutePos}\\ninput conf: ${inputConf.toFixed(3)}\\ntop1 conf: ${top1Conf.toFixed(3)}\\nid: ${tokenId}`;

      const label = document.createElement("div");
      label.className = "token-label";
      label.textContent = tokenLabel(tokenId);
      el.append(label);
      return el;
    }}

    function renderTokenRow(container, ids, frame) {{
      container.innerHTML = "";
      ids.forEach((tokenId, idx) => {{
        container.appendChild(buildTokenEl(tokenId, idx, frame));
      }});
    }}

    function fitFinalMarkdown() {{
      if (answerBox.hidden) return;

      let fontSize = __FINAL_MARKDOWN_BASE_FONT_SIZE__;
      let lineHeight = 1.08;
      let guard = 0;

      answerBox.style.fontSize = `${fontSize}px`;
      answerBox.style.lineHeight = String(lineHeight);

      while (answerBox.scrollHeight > answerBox.clientHeight && guard < 24) {{
        fontSize = Math.max(__FINAL_MARKDOWN_MIN_FONT_SIZE__, fontSize - 0.5);
        lineHeight = Math.max(0.95, lineHeight - 0.01);
        answerBox.style.fontSize = `${fontSize}px`;
        answerBox.style.lineHeight = String(lineHeight);
        guard += 1;
      }}
    }}

    function renderFrame(frameIndex) {{
      currentFrame = frameIndex;
      const frame = DATA.frames[frameIndex];
      if (!frame) return;
      const isFinalFrame = frameIndex === DATA.frames.length - 1;
      const isInitialFrame = frame.state_mode === "before_first_step";

      slider.value = String(frameIndex);
      convergenceBadge.textContent = frame.converged
        ? formatReason(frame.convergence_reason)
        : "Running";

      if (isFinalFrame) {{
        stageBox.classList.add("final-stage");
        stageTitle.textContent = "Final markdown response";
        tokensStep.hidden = true;
        tokensStep.innerHTML = "";
        stagePlaceholder.hidden = true;
        stagePlaceholder.textContent = "";
        answerBox.hidden = false;
        answerBox.innerHTML = DATA.generated_html;
        answerBox.style.fontSize = "";
        answerBox.style.lineHeight = "";
        tokenCaption.hidden = true;
        legend.hidden = true;
        requestAnimationFrame(fitFinalMarkdown);
      }} else {{
        stageBox.classList.remove("final-stage");
        stageTitle.textContent = "Visible generated tokens so far";
        answerBox.hidden = true;
        answerBox.innerHTML = "";
        tokensStep.hidden = false;
        stagePlaceholder.hidden = true;
        renderTokenRow(tokensStep, frame.visible_generated_ids, frame);
        tokenCaption.hidden = false;
        legend.hidden = false;
        tokenCaption.textContent = "Previously decoded blocks stay visible and the raw token sequence grows over time. The final generated answer is rendered as markdown only after the full sequence finishes decoding.";

        if (isInitialFrame && frame.visible_generated_ids.length === 0) {{
          tokensStep.hidden = true;
          stagePlaceholder.hidden = false;
          stagePlaceholder.textContent = "This is the initial all-mask state before the first forward pass.";
        }}
      }}
    }}

    function stopPlayback() {{
      if (timer !== null) {{
        clearInterval(timer);
        timer = null;
      }}
      playBtn.textContent = "Play";
    }}

    function togglePlayback() {{
      if (timer !== null) {{
        stopPlayback();
        return;
      }}
      playBtn.textContent = "Pause";
      timer = setInterval(() => {{
        if (currentFrame >= DATA.frames.length - 1) {{
          stopPlayback();
          return;
        }}
        renderFrame(currentFrame + 1);
      }}, __PLAYBACK_INTERVAL_MS__);
    }}

    metricFrames.textContent = Number(DATA.tpf).toFixed(2);
    metricNfe.textContent = String(DATA.nfe);
    metricGen.textContent = String(DATA.generated_token_ids.length);
    metricThreshold.textContent = Number(DATA.threshold).toFixed(2);
    promptBox.textContent = DATA.prompt;
    slider.max = String(Math.max(DATA.frames.length - 1, 0));

    renderFrame(0);

    slider.addEventListener("input", (event) => {{
      stopPlayback();
      renderFrame(Number(event.target.value));
    }});
    playBtn.addEventListener("click", togglePlayback);
    prevBtn.addEventListener("click", () => {{
      stopPlayback();
      renderFrame(Math.max(0, currentFrame - 1));
    }});
    nextBtn.addEventListener("click", () => {{
      stopPlayback();
      renderFrame(Math.min(DATA.frames.length - 1, currentFrame + 1));
    }});
    window.addEventListener("resize", () => {{
      if (!answerBox.hidden) {{
        fitFinalMarkdown();
      }}
    }});
  </script>
</body>
</html>
"""
    html = html.replace("{{", "{").replace("}}", "}")
    html = html.replace("__PAYLOAD_JSON__", payload_json)
    html = html.replace("__STAGE_SIDEBAR_WIDTH__", str(STAGE_SIDEBAR_WIDTH))
    html = html.replace("__STAGE_WIDTH__", str(STAGE_WIDTH))
    html = html.replace("__STAGE_MIN_HEIGHT__", str(STAGE_MIN_HEIGHT))
    html = html.replace("__STAGE_FINAL_HEIGHT_VH__", str(STAGE_FINAL_HEIGHT_VH))
    html = html.replace("__STAGE_FINAL_MAX_HEIGHT__", str(STAGE_FINAL_MAX_HEIGHT))
    html = html.replace("__TOKEN_GRID_MIN_HEIGHT__", str(TOKEN_GRID_MIN_HEIGHT))
    html = html.replace(
        "__FINAL_MARKDOWN_BASE_FONT_SIZE__", str(FINAL_MARKDOWN_BASE_FONT_SIZE)
    )
    html = html.replace(
        "__FINAL_MARKDOWN_MIN_FONT_SIZE__", str(FINAL_MARKDOWN_MIN_FONT_SIZE)
    )
    html = html.replace("__PLAYBACK_INTERVAL_MS__", str(PLAYBACK_INTERVAL_MS))
    return html


def main():
    args = parse_args()
    prompt = load_prompt(args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map=args.device,
    )
    model = model.to(torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    input_ids = tokenize_prompt(tokenizer, prompt)
    demo_trace, nfe, generated_tokens = model.generate_uniform_demo(
        inputs=input_ids,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        threshold=args.threshold,
    )

    generated_answer = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    demo_trace = enrich_demo_trace_for_render(demo_trace)
    token_labels = collect_token_labels(tokenizer, demo_trace)
    payload = build_demo_payload(
        prompt=prompt,
        generated_answer=generated_answer,
        demo_trace=demo_trace,
        nfe=nfe,
        token_labels=token_labels,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.write_text(render_html(payload), encoding="utf-8")

    print(f"Saved demo to: {output_path}")
    print(f"Frames: {len(demo_trace['frames'])}")
    print(f"NFE: {nfe}")
    print(f"Generated tokens: {generated_tokens.shape[1]}")


if __name__ == "__main__":
    main()
