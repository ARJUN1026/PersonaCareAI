import json
from src.agent import PersonaSupportAgent
from src.rag_pipeline import LocalRAGPipeline


def main():
    print("Persona-Adaptive Support Agent")
    print("1) Rebuild index")
    print("2) Chat")
    choice = input("Choose option [1/2]: ").strip()
    if choice == "1":
        pipeline = LocalRAGPipeline()
        count = pipeline.ingest_all(reset=True)
        print(f"Ingested {count} chunks")
        return
    agent = PersonaSupportAgent()
    print("Type 'exit' to quit.")
    while True:
        msg = input("\nYou: ").strip()
        if msg.lower() in {"exit", "quit"}:
            break
        out = agent.answer(msg)
        print(f"\nPersona: {out.persona} | Retrieval confidence: {out.retrieval_confidence} | Escalated: {out.escalated}")
        print("Sources:", json.dumps(out.retrieved_sources, indent=2))
        print("\nAgent:", out.response)
        if out.escalated:
            print("\nHandoff JSON:")
            print(json.dumps(out.handoff_summary, indent=2))

if __name__ == "__main__":
    main()
