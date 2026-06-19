from __future__ import annotations

import json
import re
from dataclasses import dataclass
from google import genai
from google.genai import types
from .config import config
from .utils import call_with_backoff

PERSONAS = ["Technical Expert", "Frustrated User", "Business Executive"]

@dataclass
class PersonaResult:
    persona: str
    confidence: float
    reasoning: str


def _rule_based_fallback(user_message: str) -> PersonaResult:
    """Safe fallback so the UI remains usable if Gemini is temporarily unavailable."""
    msg = user_message.lower()
    technical_terms = ["api", "oauth", "jwt", "bearer", "header", "logs", "endpoint", "payload", "401", "500", "database", "config", "webhook"]
    frustrated_terms = ["nothing works", "frustrated", "angry", "urgent", "immediately", "demand", "hour", "again", "!", "terrible"]
    exec_terms = ["business impact", "uptime", "operations", "timeline", "sla", "revenue", "roi", "executive", "risk"]
    scores = {
        "Technical Expert": sum(t in msg for t in technical_terms),
        "Frustrated User": sum(t in msg for t in frustrated_terms),
        "Business Executive": sum(t in msg for t in exec_terms),
    }
    persona = max(scores, key=scores.get)
    confidence = 0.55 + min(scores[persona], 4) * 0.1 if scores[persona] else 0.45
    return PersonaResult(persona=persona, confidence=min(confidence, 0.95), reasoning="Rule-based fallback classification used.")


def classify_customer_persona(user_message: str) -> PersonaResult:
    """Classify a user message into exactly one required persona using Gemini structured JSON output."""
    if not config.gemini_api_key:
        return _rule_based_fallback(user_message)

    client = genai.Client(api_key=config.gemini_api_key)
    system_instruction = (
        "You are an advanced classification engine. Analyze sentiment, vocabulary, urgency, and tone. "
        "Classify the incoming support message into exactly one persona:\n"
        "1. Technical Expert: jargon, APIs, code, configs, logs, auth, database, integrations.\n"
        "2. Frustrated User: emotional language, urgency, repeated failure, exclamation marks.\n"
        "3. Business Executive: business impact, ROI, timelines, SLA, operations, brevity.\n"
        "Return strict JSON only."
    )
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "persona": {"type": "STRING", "enum": PERSONAS},
            "confidence": {"type": "NUMBER"},
            "reasoning": {"type": "STRING"},
        },
        "required": ["persona", "confidence", "reasoning"],
    }

    try:
        response = call_with_backoff(
            client.models.generate_content,
            model=config.gemini_model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.1,
            ),
        )
        data = json.loads(response.text)
        persona = data.get("persona", "Frustrated User")
        if persona not in PERSONAS:
            persona = "Frustrated User"
        return PersonaResult(
            persona=persona,
            confidence=float(data.get("confidence", 0.6)),
            reasoning=str(data.get("reasoning", "Gemini structured classification.")),
        )
    except Exception as exc:
        fallback = _rule_based_fallback(user_message)
        fallback.reasoning += f" Gemini classification failed: {exc}"
        return fallback
