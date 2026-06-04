import httpx
import asyncio

async def test_search():
    terms = [
        "metformin chronic kidney disease",
        "metformin chronic nephropathy",
        "most significant contraindications consider prescribing metformin patient chronic",
        "metformin",
        "gestational diabetes",
        "metformin gestational diabetes"
    ]
    async with httpx.AsyncClient() as client:
        for t in terms:
            params = {
                "db": "pubmed",
                "term": t,
                "retmax": "5",
                "retmode": "json",
                "sort": "relevance",
            }
            try:
                response = await client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params)
                print(f"Term: '{t}' -> Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    idlist = data.get("esearchresult", {}).get("idlist", [])
                    print(f"  Count: {data.get('esearchresult', {}).get('count')}, IDs: {idlist}")
                else:
                    print(f"  Error: {response.text}")
            except Exception as e:
                print(f"  Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_search())
