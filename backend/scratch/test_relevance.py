import asyncio
from services.relevance import check_paper_relevance

QUERY = "Compare the efficacy and side effect profiles of SSRIs versus SNRIs in treating generalized anxiety disorder."

TEST_CASES = [
    {
        "name": "Single Concept Paper (OCD / SSRI only - Should Reject)",
        "title": "Fluvoxamine in obsessive-compulsive disorder: similar efficacy but superior tolerability in comparison with clomipramine.",
        "abstract": "This double-blind randomized clinical trial compared the efficacy and safety profile of fluvoxamine (an SSRI) with clomipramine in adult patients with obsessive-compulsive disorder (OCD). Both treatments showed similar clinical improvement.",
        "expected": False
    },
    {
        "name": "Full Concept Comparison Paper (SSRI vs SNRI in GAD - Should Accept)",
        "title": "Venlafaxine versus escitalopram in the treatment of generalized anxiety disorder: a randomized controlled trial.",
        "abstract": "We compared the efficacy, safety, and tolerability of the SNRI venlafaxine against the SSRI escitalopram in adult patients diagnosed with generalized anxiety disorder (GAD). Outcomes demonstrated similar clinical efficacy in resolving anxiety symptoms, but differed in side effect profiles (sweating and dry mouth were more prevalent in the venlafaxine group).",
        "expected": True
    }
]

async def run_tests():
    print("Starting Strict Relevance Checker Unit Tests...\n")
    success = True
    for t in TEST_CASES:
        print(f"Running Case: {t['name']}")
        print(f"Title: {t['title']}")
        relevant, reason = await check_paper_relevance(QUERY, t['title'], t['abstract'])
        print(f"Result: {relevant} | Reason: {reason}")
        if relevant == t['expected']:
            print("Status: PASS\n")
        else:
            print(f"Status: FAIL (Expected {t['expected']}, got {relevant})\n")
            success = False
            
    if success:
        print("All relevance checker unit tests passed successfully!")
    else:
        print("Some relevance checker unit tests failed.")

if __name__ == "__main__":
    asyncio.run(run_tests())
