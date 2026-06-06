import httpx
import sys

# Reconfigure stdout to support UTF-8 on Windows consoles to prevent charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://127.0.0.1:8080/query"
QUERY = 'Meta-analysis evidence for the efficacy of vitamin D supplementation in preventing falls in elderly populations."'

def test_query():
    print(f"Sending query: '{QUERY}'\n")
    with httpx.Client(timeout=60.0) as client:
        try:
            response = client.post(URL, json={"query": QUERY})
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print("\nAnswer:")
                print(data.get("answer"))
                print(f"\nConfidence Label: {data.get('confidence_label')}")
                print(f"Confidence Note: {data.get('confidence_note')}")
                print(f"Confidence Score: {data.get('confidence_score')}")
                print("\nCitations:")
                for i, c in enumerate(data.get("citations", []), 1):
                    print(f"{i}. {c.get('title')} (PMID: {c.get('pmid')})")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_query()
