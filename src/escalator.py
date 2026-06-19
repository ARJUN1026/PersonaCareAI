from __future__ import annotations

import json
from dataclasses import dataclass, field
from .config import config

@dataclass
class ConversationTurn:
    role: str
    content: str

@dataclass
class EscalationDecision:
    escalated: bool
    reason: str | None = None
    handoff_summary: dict | None = None

@dataclass
class ConversationState:
    history: list[ConversationTurn] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.history.append(ConversationTurn(role=role, content=content))

    def dissatisfied_count(self) -> int:
        count = 0
        for turn in self.history:
            if turn.role != "user":
                continue
            msg = turn.content.lower()
            if any(keyword in msg for keyword in config.negative_keywords):
                count += 1
        return count


def is_sensitive_issue(user_query: str) -> str | None:
    msg = user_query.lower()
    for keyword in config.sensitive_keywords:
        if keyword in msg:
            return keyword
    return None


def build_handoff_summary(
    user_query: str,
    persona: str,
    persona_reasoning: str,
    context_chunks: list[dict],
    reason: str,
    history: list[ConversationTurn],
) -> dict:
    best_score = max([c["score"] for c in context_chunks], default=0.0)
    return {
        "persona": persona,
        "persona_reasoning": persona_reasoning,
        "detected_issue": user_query,
        "conversation_history": [{"role": t.role, "content": t.content} for t in history[-10:]],
        "retrieved_sources": [
            {
                "source": c.get("source"),
                "page_or_section": c.get("page"),
                "chunk_index": c.get("chunk_index"),
                "score": round(float(c.get("score", 0.0)), 3),
            }
            for c in context_chunks
        ],
        "confidence_score": round(float(best_score), 3),
        "escalation_reason": reason,
        "attempted_steps": [
            t.content[:220] for t in history if t.role == "assistant" and any(x in t.content.lower() for x in ["try", "check", "verify", "reset", "step"])
        ][-5:],
        "recommended_action": "Route to the correct human support queue. Verify account/system state, review retrieved sources, and contact the user with a documented next step.",
    }


def check_escalation(user_query: str, persona: str, persona_reasoning: str, context_chunks: list[dict], state: ConversationState) -> EscalationDecision:
    best_score = max([c["score"] for c in context_chunks], default=0.0)
    reason = None
    if not context_chunks:
        reason = "No relevant knowledge base chunks were retrieved."
    elif best_score < config.confidence_threshold:
        reason = f"Low retrieval confidence: {best_score:.2f} < {config.confidence_threshold:.2f}."
    else:
        sensitive_keyword = is_sensitive_issue(user_query)
        if sensitive_keyword:
            reason = f"Sensitive issue detected: {sensitive_keyword}."
        elif state.dissatisfied_count() >= config.escalate_after_dissatisfied:
            reason = "Repeated dissatisfaction detected across multiple turns."
        elif any(x in user_query.lower() for x in ["human agent", "talk to human", "escalate", "manager"]):
            reason = "User explicitly requested human support."

    if reason:
        return EscalationDecision(
            escalated=True,
            reason=reason,
            handoff_summary=build_handoff_summary(user_query, persona, persona_reasoning, context_chunks, reason, state.history + [ConversationTurn("user", user_query)]),
        )
    return EscalationDecision(escalated=False)


def handoff_json(summary: dict | None) -> str:
    return json.dumps(summary or {}, indent=2, ensure_ascii=False)
