import httpx
import json
import sys

# Reconfigure stdout to support UTF-8 on Windows consoles to prevent charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://127.0.0.1:8080/query/stream"
QUERY = "What are the latest clinical trial outcomes for metformin in treating gestational diabetes?"

def test_streaming_query():
    print(f"Sending streaming query: '{QUERY}'\n")
    
    payload = {"query": QUERY}
    
    with httpx.Client(timeout=60.0) as client:
        try:
            with client.stream("POST", URL, json=payload) as response:
                print(f"Status Code: {response.status_code}")
                print(f"Headers: {response.headers.get('content-type')}\n")
                
                if response.status_code != 200:
                    print(f"Error: {response.read().decode('utf-8')}")
                    return

                current_event = ""
                print("Answer Stream: ", end="", flush=True)
                
                for line in response.iter_lines():
                    trimmed = line.strip()
                    if not trimmed:
                        continue
                    
                    if trimmed.startswith("event:"):
                        current_event = trimmed.split(":", 1)[1].strip()
                    elif trimmed.startswith("data:"):
                        raw_data = trimmed.split(":", 1)[1].strip()
                        if raw_data == "[DONE]":
                            continue
                        try:
                            parsed_data = json.loads(raw_data)
                            if current_event == "token":
                                print(parsed_data, end="", flush=True)
                            elif current_event == "metadata":
                                print("\n\n--- Metadata Event ---")
                                print(f"Confidence Label: {parsed_data.get('confidence_label')}")
                                print(f"Confidence Note: {parsed_data.get('confidence_note')}")
                                print(f"Confidence Score: {parsed_data.get('confidence_score')}")
                                print("\nCitations:")
                                for i, c in enumerate(parsed_data.get("citations", []), 1):
                                    print(f"{i}. {c.get('title')} (PMID: {c.get('pmid')})")
                            elif current_event == "error":
                                print(f"\n[STREAM ERROR] {parsed_data}")
                        except Exception as e:
                            print(f"\nFailed to parse SSE data: {e} | data={raw_data}")
                            
        except Exception as e:
            print(f"\nConnection Error: {e}")

if __name__ == "__main__":
    test_streaming_query()
