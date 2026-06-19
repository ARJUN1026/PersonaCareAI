import json
import streamlit as st
from src.agent import PersonaSupportAgent
from src.config import config
from src.rag_pipeline import LocalRAGPipeline

st.set_page_config(page_title="Persona Support Agent", page_icon="🤖", layout="wide")

if not config.has_gemini_api_key:
    st.warning(
        "GEMINI_API_KEY is missing or the placeholder value is still present. "
        "Copy .env.example to .env, add your actual Gemini key, and restart the app."
    )

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1180px;}
.hero {padding: 1.2rem 1.4rem; border-radius: 24px; background: linear-gradient(135deg,#eef4ff,#ffffff); border:1px solid #e7edf8; margin-bottom:1rem; color: #0f172a;}
.hero h1 { margin: 0 0 0.35rem; color: inherit; }
.hero p { margin: 0; color: #475569; }
.pill {display:inline-block; padding: .35rem .7rem; border-radius: 999px; background:#eef4ff; margin:.15rem; font-size:.86rem;}
.alert {padding:1rem; border-radius:18px; background:#fff1f2; border:1px solid #fecdd3;}
.small {font-size:.88rem; color:#64748b;}
@media (prefers-color-scheme: dark) {
    .hero { background: rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); color: #f8fafc; }
    .hero p, .small { color: #cbd5e1; }
    .pill { background: rgba(255,255,255,0.12); color: #f8fafc; border: 1px solid rgba(255,255,255,0.18); }
    .alert { background: rgba(248,113,113,0.15); border-color: rgba(248,113,113,0.35); color: #f8fafc; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🤖 Persona-Adaptive Customer Support Agent</h1>
<p class="small">Gemini + ChromaDB RAG + persona-aware prompting + human escalation handoff.</p>
</div>
""", unsafe_allow_html=True)

if "agent" not in st.session_state:
    with st.spinner("Loading agent and ChromaDB collection..."):
        st.session_state.agent = PersonaSupportAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Controls")
    if not config.has_gemini_api_key:
        st.warning(
            "GEMINI_API_KEY is missing or still a placeholder. "
            "Re-ingest is disabled until you add a valid key to .env."
        )
    if st.button(
        "🔄 Re-ingest knowledge base",
        use_container_width=True,
        disabled=not config.has_gemini_api_key,
    ):
        with st.spinner("Embedding documents with Gemini and rebuilding ChromaDB..."):
            pipeline = LocalRAGPipeline()
            count = pipeline.ingest_all(reset=True)
            st.session_state.agent = PersonaSupportAgent()
            st.success(f"Ingested {count} chunks")
    st.divider()
    st.subheader("Demo scenarios")
    examples = [
        "Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!",
        "What are the header parameter requirements for your bearer token auth implementation?",
        "Our operational uptime is decreasing. We need a timeline of when billing disputes are resolved.",
        "I'm experiencing an issue with your database integration that's causing internal errors.",
        "My billing statement has unexpected duplicate charges. I demand an immediate refund!",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_prompt = ex

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a support question...") or st.session_state.pop("pending_prompt", None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Classifying persona, retrieving context, checking escalation..."):
            try:
                out = st.session_state.agent.answer(prompt)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Persona", out.persona)
                c2.metric("Persona confidence", out.persona_confidence)
                c3.metric("Retrieval confidence", out.retrieval_confidence)
                c4.metric("Escalated", "Yes" if out.escalated else "No")
                st.caption(out.persona_reasoning)
                st.markdown(out.response)

                st.subheader("Retrieved sources")
                for src in out.retrieved_sources:
                    st.markdown(
                        f"<span class='pill'>{src['source']} · {src['page_or_section']} · chunk {src['chunk_index']} · score {src['score']}</span>",
                        unsafe_allow_html=True,
                    )

                if out.escalated:
                    st.markdown(f"<div class='alert'><b>Human escalation required:</b> {out.escalation_reason}</div>", unsafe_allow_html=True)
                    st.subheader("Handoff JSON")
                    st.json(out.handoff_summary)
                st.session_state.messages.append({"role": "assistant", "content": out.response})
            except Exception as exc:
                st.error(str(exc))
                st.info("Check that GEMINI_API_KEY is set in .env and run: python scripts/ingest.py")
