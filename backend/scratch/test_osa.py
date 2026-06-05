import httpx

URL = "http://127.0.0.1:8080/query"
QUERY = "What is the evidence-based association between obesity (BMI > 30) and obstructive sleep apnea (OSA)?"

def test_query():
    print(f"Sending query: '{QUERY}'\n")
    with httpx.Client(timeout=60.0) as client:
        try:
            response = client.post(URL, json={"query": QUERY})
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print("\nAnswer:")
                ans = data.get("answer", "")
                print(ans.encode("ascii", errors="ignore").decode("ascii"))
                print(f"\nConfidence Label: {data.get('confidence_label')}")
                print(f"Confidence Note: {data.get('confidence_note')}")
                print(f"Confidence Score: {data.get('confidence_score')}")
                print("\nCitations:")
                citations = data.get("citations", [])
                for i, c in enumerate(citations, 1):
                    title = c.get('title', '')
                    print(f"{i}. {title.encode('ascii', 'ignore').decode('ascii')} (PMID: {c.get('pmid')})")
                
                # Assertions for verification
                assert data.get("confidence_score", 0) > 0.40, f"Expected confidence score > 0.40, got {data.get('confidence_score')}"
                assert len(citations) > 0, "Expected at least one citation"
                print("\nVerification Status: PASS - Valid results and high confidence retrieved.")
            else:
                print(f"Error: {response.text}")
                print("\nVerification Status: FAIL")
        except Exception as e:
            print(f"Connection/Assertion Error: {e}")
            print("\nVerification Status: FAIL")

if __name__ == "__main__":
    test_query()
