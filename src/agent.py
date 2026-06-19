from __future__ import annotations

from dataclasses import dataclass
from .classifier import classify_customer_persona
from .rag_pipeline import LocalRAGPipeline
from .generator import generate_adaptive_response, escalation_response
from .escalator import ConversationState, check_escalation
from .config import config

@dataclass
class AgentOutput:
    user_message: str
    persona: str
    persona_confidence: float
    persona_reasoning: str
    retrieved_sources: list[dict]
    retrieval_confidence: float
    response: str
    escalated: bool
    escalation_reason: str | None
    handoff_summary: dict | None

class PersonaSupportAgent:
    def __init__(self):
        self.state = ConversationState()
        self.rag = LocalRAGPipeline()

    def answer(self, user_message: str) -> AgentOutput:
        persona_result = classify_customer_persona(user_message)
        context_chunks = self.rag.retrieve_context(user_message, top_k=config.top_k)
        retrieval_confidence = max([c["score"] for c in context_chunks], default=0.0)
        decision = check_escalation(
            user_message,
            persona_result.persona,
            persona_result.reasoning,
            context_chunks,
            self.state,
        )
        if decision.escalated:
            response = escalation_response(persona_result.persona, decision.reason or "Escalation policy triggered")
        else:
            response = generate_adaptive_response(user_message, persona_result.persona, context_chunks)

        self.state.add("user", user_message)
        self.state.add("assistant", response)
        return AgentOutput(
            user_message=user_message,
            persona=persona_result.persona,
            persona_confidence=round(persona_result.confidence, 3),
            persona_reasoning=persona_result.reasoning,
            retrieved_sources=[
                {"source": c["source"], "page_or_section": c.get("page"), "chunk_index": c.get("chunk_index"), "score": round(c["score"], 3)}
                for c in context_chunks
            ],
            retrieval_confidence=round(retrieval_confidence, 3),
            response=response,
            escalated=decision.escalated,
            escalation_reason=decision.reason,
            handoff_summary=decision.handoff_summary,
        )
