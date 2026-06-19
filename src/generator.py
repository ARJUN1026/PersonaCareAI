from __future__ import annotations

from google import genai
from google.genai import types
from .config import config
from .utils import call_with_backoff


def persona_instructions(persona: str) -> str:
    if persona == "Technical Expert":
        return (
            "You are a Senior Systems Engineer. Provide precise root-cause analysis, "
            "configuration checks, HTTP/API details, logs to inspect, and step-by-step diagnostics. "
            "Use code blocks only when the retrieved context supports them."
        )
    if persona == "Frustrated User":
        return (
            "You are a deeply empathetic Customer Care Specialist. Start by validating the inconvenience. "
            "Use simple language, short bullet points, and reassuring action-oriented steps. Avoid jargon."
        )
    return (
        "You are a concise Client Relations Director. Focus on business impact, risk, next update, and resolution guidance. "
        "Keep it brief, professional, and avoid unnecessary technical detail."
    )


def generate_adaptive_response(user_query: str, persona: str, context_chunks: list[dict]) -> str:
    if not config.has_gemini_api_key:
        return (
            "GEMINI_API_KEY is required to generate responses. "
            "Copy .env.example to .env and add your actual Gemini key, then restart the app."
        )

    context_text = "\n\n".join(
        f"Source [{c['source']} | page/section {c.get('page')} | score {c['score']:.3f}]:\n{c['text']}"
        for c in context_chunks
    )
    full_system_prompt = (
        f"{persona_instructions(persona)}\n\n"
        "CRITICAL RULES:\n"
        "- Base your response ONLY on the provided factual context documents.\n"
        "- If the answer is not present, say the documentation does not confirm it and recommend escalation.\n"
        "- Do not invent prices, legal claims, refund promises, account details, or resolution times.\n"
        "- End with a compact 'Sources used' line.\n\n"
        f"FACTUAL CONTEXT DOCUMENTS:\n{context_text or 'NO_RELEVANT_CONTEXT'}"
    )
    client = genai.Client(api_key=config.gemini_api_key)
    response = call_with_backoff(
        client.models.generate_content,
        model=config.gemini_model,
        contents=user_query,
        config=types.GenerateContentConfig(system_instruction=full_system_prompt, temperature=0.2),
    )
    return response.text.strip()


def escalation_response(persona: str, reason: str) -> str:
    if persona == "Frustrated User":
        return (
            "I understand how frustrating this is, and I do not want to give you an uncertain answer. "
            f"This needs a human specialist because: {reason} I have prepared a handoff summary so you do not have to repeat everything."
        )
    if persona == "Business Executive":
        return (
            f"This should be escalated to a human specialist. Reason: {reason} "
            "A structured handoff summary is ready for faster triage and follow-up."
        )
    return (
        f"Escalation recommended. Reason: {reason} "
        "The handoff JSON includes persona, retrieved sources, confidence, and recommended next technical checks."
    )
