import httpx
import json
import time

URL = "http://127.0.0.1:8080/query"

TEST_QUERIES = [
    ("What are the latest clinical trial outcomes for metformin in treating gestational diabetes?", "Medical Research"),
    ("What should I prescribe for this patient who has high blood pressure?", "Patient Advice"),
    ("What is the weather like in New York?", "Non-medical"),
    ("Hello there!", "Greeting"),
]

def run_api_tests():
    print("Starting API integration tests against running server...\n")
    
    with httpx.Client(timeout=30.0) as client:
        for query, label in TEST_QUERIES:
            print(f"Query: '{query}' ({label})")
            payload = {"query": query}
            try:
                start = time.time()
                response = client.post(URL, json=payload)
                elapsed = time.time() - start
                print(f"  Status Code: {response.status_code} (took {elapsed:.2f}s)")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"  Answer: {data.get('answer')}")
                    print(f"  Confidence Label: {data.get('confidence_label')}")
                    print(f"  Confidence Note: {data.get('confidence_note')}")
                    print(f"  Citations Count: {len(data.get('citations', []))}")
                else:
                    print(f"  Error Response: {response.text}")
            except Exception as e:
                print(f"  Connection Error: {e}")
            print("-" * 50)
            time.sleep(2)  # Avoid rate limiting

if __name__ == "__main__":
    run_api_tests()
