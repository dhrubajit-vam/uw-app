"""
AI narrative layer - Azure OpenAI (gpt-4o) turns the deterministic scoring,
explainability, and recommendation output into polished, plain-English
underwriter narrative. It NEVER decides anything: every number, band,
premium, reason code, and alternative shown elsewhere in the app is
computed by the rules engine / models before this module ever runs. The
model is instructed to only phrase and prioritize those given facts, not
invent new ones - if the API is unreachable, misconfigured, or errors out
for any reason, every function here returns None and callers fall back to
the plain rule-based text that already existed. A live demo should never
break because an AI call failed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_TIMEOUT_SECONDS = 20
_TEMPERATURE = 0.3
_MAX_TOKENS = 260

_client = None
_client_init_attempted = False


def _config_value(name: str) -> Optional[str]:
    """Check the environment first (.env locally), then Streamlit's own
    Secrets manager (st.secrets) - that's how this gets configured on
    Streamlit Community Cloud, where .env doesn't exist (it's git-ignored
    on purpose, so the API key never ends up in the repo)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _get_client():
    """Lazily build the Azure OpenAI client. Returns None (not an
    exception) if credentials are missing/invalid or the package can't be
    imported, so every caller can just check for None and fall back."""
    global _client, _client_init_attempted
    if _client_init_attempted:
        return _client
    _client_init_attempted = True
    try:
        from openai import AzureOpenAI
        api_key = _config_value("AZURE_OPENAI_API_KEY")
        endpoint = _config_value("AZURE_OPENAI_ENDPOINT")
        api_version = _config_value("AZURE_OPENAI_API_VERSION")
        if not (api_key and endpoint and api_version):
            return None
        _client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version,
                               timeout=_TIMEOUT_SECONDS)
    except Exception:
        _client = None
    return _client


def is_available() -> bool:
    return _get_client() is not None


def _strip_dashes(text: str) -> str:
    """Belt-and-braces on the prompt instruction: em/en-dashes read as
    machine-written, so rewrite any that slip through into ordinary
    punctuation. A dash used as an aside ('x - y') becomes a comma; one
    used as a connector between clauses becomes a colon."""
    text = text.replace(" — ", ", ").replace(" – ", ", ")
    text = text.replace("—", ", ").replace("–", ", ")
    return text.replace(" ,", ",").replace(",,", ",")


def _complete(system_prompt: str, user_prompt: str) -> Optional[str]:
    client = _get_client()
    if client is None:
        return None
    deployment = _config_value("AZURE_OPENAI_DEPLOYMENT_NAME") or "gpt-4o"
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _strip_dashes(text) or None
    except Exception:
        # Network hiccup, quota, bad deployment name, etc. - never let a
        # live demo crash on this; the caller falls back to rule-based text.
        return None


_GROUNDING_RULES = (
    "You are writing for a licensed underwriter reviewing a real submission. "
    "You are given exact figures already computed by a deterministic rules "
    "engine and machine learning models - never invent, guess, or round "
    "loosely beyond what is given. Do not introduce any fact, number, "
    "reason, or recommendation that isn't explicitly provided below. Do not "
    "mention that you are an AI, a language model, or GPT. Do not use "
    "bullet points or headers - write flowing, professional prose only. "
    "Never use em-dashes or en-dashes; use commas, colons, or separate "
    "sentences instead. Avoid stock openers like 'Based on the analysis' "
    "or 'This submission has been assessed'; lead with the substance. "
    "Keep it to 3-5 sentences."
)


def narrate_decision(result, explanation) -> Optional[str]:
    """A polished paragraph explaining the underwriting decision - grounded
    strictly in result (rules_engine.ScoringResult) and explanation
    (scoring.explain.Explanation)."""
    facts = [
        f"UW risk band: {result.uw_risk_band}",
        f"Recommended action: {result.recommended_action}",
        f"Composite UW score: {result.composite_score}/100",
        f"Loss propensity score: {result.loss_propensity_score:.1f}/100 (higher = worse risk)",
        f"Bind propensity score: {result.bind_propensity_score:.1f}/100 (higher = more likely to bind)",
        f"Appetite/eligibility score: {result.appetite_score:.1f}/100",
        f"Technical premium: ${result.technical_premium:,.0f}",
        f"Final quoted premium: ${result.final_quoted_premium:,.0f}",
        f"Decision confidence: {explanation.confidence}",
        f"Manual review required: {'yes' if explanation.manual_review else 'no'}",
    ]
    if result.hard_stop:
        facts.append(f"Hard-stop reason(s): {'; '.join(result.hard_stop_reasons)}")
    if explanation.appetite_reasons:
        facts.append("Top appetite/eligibility penalties: " + "; ".join(
            f"{r.label} ({r.weight:.0f}/100 relative weight)" for r in explanation.appetite_reasons[:3]))
    if explanation.loss_reasons:
        facts.append("Top loss-cost drivers: " + "; ".join(
            f"{r.label} ({r.direction} loss cost, {r.weight:.0f}/100 relative weight)"
            for r in explanation.loss_reasons[:3]))
    if explanation.bind_reasons:
        facts.append("Top bind-likelihood drivers: " + "; ".join(
            f"{r.label} ({r.direction} bind likelihood, {r.weight:.0f}/100 relative weight)"
            for r in explanation.bind_reasons[:3]))

    user_prompt = (
        "Write the underwriting decision narrative for this submission using only these facts:\n"
        + "\n".join(f"- {f}" for f in facts)
    )
    return _complete(_GROUNDING_RULES, user_prompt)


def narrate_alternatives(result, alternatives) -> Optional[str]:
    """A polished paragraph synthesizing the underwriter alternatives to a
    flat decline - grounded strictly in result and the Alternative list
    from scoring.advisor.build_alternatives."""
    if not alternatives:
        return None
    facts = [f"Current outcome: {result.uw_risk_band} band, recommended action '{result.recommended_action}'."]
    for alt in alternatives:
        line = f"[{alt.category}] {alt.title}: {alt.change_summary}"
        if alt.result is not None:
            line += (f" If applied: composite score {result.composite_score:.1f} -> "
                     f"{alt.result.composite_score:.1f}, band {result.uw_risk_band} -> "
                     f"{alt.result.uw_risk_band}, premium ${result.final_quoted_premium:,.0f} -> "
                     f"${alt.result.final_quoted_premium:,.0f}.")
        if alt.note:
            line += f" Note: {alt.note}"
        facts.append(line)

    user_prompt = (
        "An underwriter is deciding what to do instead of a flat decline. Using ONLY the "
        "alternatives below (already computed and re-scored by the rules engine - do not "
        "add options that aren't listed), write a short professional recommendation "
        "explaining which alternative(s) are most worth pursuing and why:\n"
        + "\n".join(f"- {f}" for f in facts)
    )
    return _complete(_GROUNDING_RULES, user_prompt)
