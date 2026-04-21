import os
import re
import json
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL   = "google/gemma-3-27b-it"

EVAL_CATEGORIES = [
    "Completeness",
    "Clarity",
    "Technical Accuracy",
    "Getting Started / Usability",
    "Formatting & Structure",
]

REQUIRED_CATEGORY_KEYS = {
    "name", "generated_score", "original_score",
    "generated_strengths", "generated_weaknesses",
    "original_strengths",  "original_weaknesses",
}

# ── NVIDIA NIM call ──────────────────────────────────────────────────────────

def call_nvidia(prompt: str) -> str:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY not found in environment / .env")

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.7,
        "stream": False,
    }
    headers = {
        "authorization": f"Bearer {api_key}",
        "accept": "application/json",
        "content-type": "application/json",
    }

    response = requests.post(NVIDIA_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    if "choices" not in data:
        raise ValueError(f"Unexpected API response shape: {data}")
    return data["choices"][0]["message"]["content"]


# ── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt(generated: str, original: str) -> str:
    categories_list = "\n".join(f"- {c}" for c in EVAL_CATEGORIES)
    return f"""
You are an expert technical writer evaluating two README files for the same software project.

README A is AI-generated. README B is the original human-written README.

Your task: evaluate both READMEs across the following categories:
{categories_list}

For each category, provide:
- A score out of 10 for each README
- A short strength of README A (generated)
- A short weakness of README A (generated)
- A short strength of README B (original)
- A short weakness of README B (original)

Then provide a brief overall verdict comparing the two.

Return ONLY a valid JSON object — no markdown fences, no preamble, no explanation outside the JSON.

Schema:
{{
  "categories": [
    {{
      "name": "<category name>",
      "generated_score": <integer 1-10>,
      "original_score": <integer 1-10>,
      "generated_strengths": "<string>",
      "generated_weaknesses": "<string>",
      "original_strengths": "<string>",
      "original_weaknesses": "<string>"
    }}
  ],
  "overall_verdict": "<string>"
}}

---
README A (AI-Generated):
{generated}

---
README B (Original):
{original}
""".strip()


# ── Response parser & validator ──────────────────────────────────────────────

def strip_fences(text: str) -> str:
    """Remove markdown code fences the LLM might add despite instructions."""
    return re.sub(r"```(?:json)?(.*?)```", r"\1", text, flags=re.DOTALL).strip()


def parse_and_validate(raw: str) -> dict:
    cleaned = strip_fences(raw)
    data = json.loads(cleaned)                      # raises json.JSONDecodeError if malformed

    # Top-level shape check
    if "categories" not in data or "overall_verdict" not in data:
        raise ValueError("Response missing 'categories' or 'overall_verdict' keys.")

    # Per-category shape check
    for i, cat in enumerate(data["categories"]):
        missing = REQUIRED_CATEGORY_KEYS - cat.keys()
        if missing:
            raise ValueError(f"Category {i} ('{cat.get('name', '?')}') missing keys: {missing}")

    return data


# ── Streamlit UI ─────────────────────────────────────────────────────────────

def score_color(score: int) -> str:
    if score >= 8:
        return "#2ecc71"
    elif score >= 5:
        return "#f39c12"
    return "#e74c3c"


def render_score_badge(label: str, score: int):
    color = score_color(score)
    st.markdown(
        f"""
        <div style="display:inline-flex;align-items:center;gap:8px;margin-bottom:4px">
            <span style="font-weight:600;font-size:0.9rem">{label}</span>
            <span style="background:{color};color:#fff;padding:2px 10px;
                         border-radius:12px;font-weight:700;font-size:0.95rem">
                {score}/10
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(data: dict):
    st.markdown("---")
    st.subheader("📊 Evaluation Results")

    # ── Score summary table ──────────────────────────────────────────────────
    st.markdown("#### Score Summary")
    cols = st.columns([3, 1, 1])
    cols[0].markdown("**Category**")
    cols[1].markdown("**Generated**")
    cols[2].markdown("**Original**")

    for cat in data["categories"]:
        c0, c1, c2 = st.columns([3, 1, 1])
        c0.write(cat["name"])
        c1.markdown(
            f"<span style='color:{score_color(cat['generated_score'])};font-weight:700'>"
            f"{cat['generated_score']}/10</span>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<span style='color:{score_color(cat['original_score'])};font-weight:700'>"
            f"{cat['original_score']}/10</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Per-category deep dive ───────────────────────────────────────────────
    st.markdown("#### Detailed Breakdown")
    for cat in data["categories"]:
        with st.expander(f"**{cat['name']}**  —  Generated: {cat['generated_score']}/10  |  Original: {cat['original_score']}/10"):
            left, right = st.columns(2)

            with left:
                st.markdown("**AI-Generated README**")
                render_score_badge("Score", cat["generated_score"])
                st.markdown(f"**Strength:** {cat['generated_strengths']}")
                st.markdown(f"**Weakness:** {cat['generated_weaknesses']}")

            with right:
                st.markdown("**Original README**")
                render_score_badge("Score", cat["original_score"])
                st.markdown(f"**Strength:** {cat['original_strengths']}")
                st.markdown(f"**Weakness:** {cat['original_weaknesses']}")

    st.markdown("---")

    # ── Overall verdict ──────────────────────────────────────────────────────
    st.markdown("#### 🏁 Overall Verdict")
    st.info(data["overall_verdict"])


def main():
    st.set_page_config(
        page_title="README Evaluator",
        page_icon="🔍",
        layout="wide",
    )

    st.title("README Evaluator")
    st.write(
        "Paste both READMEs below. The LLM will score and compare them "
        "across key quality dimensions."
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("AI-Generated README")
        generated_readme = st.text_area(
            label="Paste the generated README here",
            placeholder="Paste your AI-generated README.md content...",
            height=400,
            label_visibility="collapsed",
        )

    with right_col:
        st.subheader("Original README")
        original_readme = st.text_area(
            label="Paste the original README here",
            placeholder="Paste the original repo README.md content...",
            height=400,
            label_visibility="collapsed",
        )

    if st.button("⚡ Evaluate READMEs", type="primary", use_container_width=True):
        if not generated_readme.strip() or not original_readme.strip():
            st.warning("Both fields must be filled before evaluating.")
            st.stop()

        try:
            with st.spinner("Calling NVIDIA NIM and evaluating..."):
                prompt   = build_prompt(generated_readme, original_readme)
                raw      = call_nvidia(prompt)
                results  = parse_and_validate(raw)

            render_results(results)

        except json.JSONDecodeError as e:
            st.error(f"Failed to parse LLM response as JSON: {e}")
            with st.expander("Raw LLM response"):
                st.code(raw)

        except ValueError as e:
            st.error(f"Validation error: {e}")
            with st.expander("Raw LLM response"):
                st.code(raw)

        except Exception as e:
            st.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()