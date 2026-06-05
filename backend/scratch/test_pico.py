import time
import httpx

URL = "http://127.0.0.1:8080/query"
TEST_QUERIES = [
    "My blood pressure was 150/95 at home. Is that considered high? What medication usually fixes that?",
    "I've had a headache for three days and my vision is a bit blurry. Is this something I need to worry about?",
    "i am 45 years old, i am having high bp and heart rate"
]

def run_tests():
    print("Starting PICO Conversational Translation Verification...\n")
    success = True
    
    with httpx.Client(timeout=60.0) as client:
        for idx, query in enumerate(TEST_QUERIES, 1):
            print(f"[{idx}] Sending Query: '{query}'")
            try:
                response = client.post(URL, json={"query": query})
                print(f"Status Code: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    ans = data.get("answer", "")
                    print(f"Answer Preview: {ans[:200].encode('ascii', errors='ignore').decode('ascii')}...")
                    print(f"Confidence Label: {data.get('confidence_label')}")
                    print(f"Confidence Note: {data.get('confidence_note')}")
                    print(f"Confidence Score: {data.get('confidence_score')}")
                    citations = data.get("citations", [])
                    print(f"Citations Retrieved: {len(citations)}")
                    for c in citations[:2]:
                        title = c.get('title', '')
                        print(f"  - {title.encode('ascii', 'ignore').decode('ascii')} (PMID: {c.get('pmid')})")
                    
                    # Assertions for verification
                    if data.get("confidence_score", 0) <= 0.40:
                        print("Status: FAIL (Confidence score too low)")
                        success = False
                    elif len(citations) == 0:
                        print("Status: FAIL (Zero citations retrieved)")
                        success = False
                    else:
                        print("Status: PASS\n")
                else:
                    print(f"Error: {response.text}")
                    print("Status: FAIL\n")
                    success = False
            except Exception as e:
                print(f"Connection/Assertion Error: {e}")
                print("Status: FAIL\n")
                success = False
                
            # Pace requests to avoid rate limits
            if idx < len(TEST_QUERIES):
                print(f"Sleeping 25 seconds to respect rate limits...\n")
                time.sleep(25)
                
    if success:
        print("All PICO query translation verification tests passed successfully!")
    else:
        print("Some PICO query translation verification tests failed.")

if __name__ == "__main__":
    run_tests()
