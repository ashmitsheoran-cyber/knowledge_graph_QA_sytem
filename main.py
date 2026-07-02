import json
from agent_engine import agent_app, classify_intent, handle_chat, extract_intent

LOG_FILE = "eval_log.jsonl"

def log_interaction(query, answer, context, gt="N/A"):
    entry = {
        "question": query,
        "answer": answer,
        "contexts": context,
        "ground_truth": gt
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def start_qa():
    print("\n" + "="*60)
    print("HYBRID AGENTIC RAG SYSTEM ONLINE")
    print("="*60)

    while True:
        query = input("\nAsk a question: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        if not query:
            continue

        try:
            if classify_intent(query) == "CHAT":
                intent_data = extract_intent(query)
                print("\n" + "—"*60)
                print(handle_chat(query, intent_data.get("tone", "neutral")))
                print("—"*60)
                continue

            initial_state = {
                "query":          query,
                "query_en":       "",
                "lang":           "en",
                "clean_question": "",
                "search_pref":    "no_preference",
                "format_pref":    "no_preference",
                "tone":           "neutral",
                "doc_filter":     "",
                "version_pref":   "",
                "matched_doc":    "",
                "history":        [],
                "grouped_docs":   "{}",
                "context":        [],
                "answer":         "",
                "source_type":    "",
                "iterations":     0,
                "gap":            ""
            }

            result = agent_app.invoke(initial_state)

            print("\n" + "—"*60)
            print(result['answer'])
            print("—"*60)

            if result.get('source_type') == "Web":
                gt = input("\nWeb Data Used. Enter Ground Truth for eval (or Enter to skip): ").strip()
                log_interaction(query, result['answer'], result['context'], gt or "N/A")
            else:
                print("\nAnswered directly from Local Vector Memory.")
                log_interaction(query, result['answer'], result['context'], "N/A")

        except Exception as e:
            print(f"System Error: {str(e)}")

if __name__ == "__main__":
    start_qa()