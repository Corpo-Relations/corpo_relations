import sys
import json
import os

# Add current directory to path so we can import Agent and main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import main
    import agentv2 as agent
    import sentiment_analysis
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    IMPORT_ERROR = str(e)

def main_loop():
    if not TRANSFORMERS_AVAILABLE:
        print(json.dumps({"status": "error", "message": f"Server import error: {IMPORT_ERROR}"}), flush=True)
        return

    # Pre-load the model
    try:
        main.get_text_generation_model()
        print("READY", flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Model load error: {str(e)}"}), flush=True)
        return

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            command = data.get("command")
            
            if command == "generate":
                history = data.get("history")
                response = main.generate_response(history)
                print(json.dumps({"status": "ok", "response": response}), flush=True)
            
            elif command == "summarize":
                text = data.get("text")
                summary = main.summarize_memory(text)
                print(json.dumps({"status": "ok", "summary": summary}), flush=True)
            
            elif command == "sentiment":
                text = data.get("text")
                label, score = sentiment_analysis.get_sentiment(text)
                print(json.dumps({"status": "ok", "label": label, "score": score}), flush=True)
            
            elif command == "ping":
                print(json.dumps({"status": "ok", "response": "pong"}), flush=True)
            
            elif command == "exit":
                break
            else:
                print(json.dumps({"status": "error", "message": "Unknown command"}), flush=True)
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}), flush=True)

if __name__ == "__main__":
    main_loop()
