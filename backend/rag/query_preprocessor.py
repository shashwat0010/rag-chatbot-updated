import re
from typing import List

from rag.symptom_mapper import infer_diseases

CASUAL_TO_MEDICAL = {
    "fat": "obesity",
    "overweight": "obesity",
    "obese": "obesity",
    "high sugar": "hyperglycemia",
    "frequent urination": "polyuria",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "thirst": "polydipsia",
    "excessive thirst": "polydipsia",
    "high blood pressure": "hypertension",
    "low blood pressure": "hypotension",
    "kidney pain": "nephropathy",
    "kidney disease": "nephropathy",
    "nerve pain": "neuropathy",
    "eye damage": "retinopathy",
    "vision loss": "retinopathy",
    "swelling": "edema",
    "fever": "pyrexia",
    "runny nose": "rhinitis",
    "heart attack": "myocardial infarction",
    "stroke": "cerebrovascular accident",
    "shortness of breath": "dyspnea",
    "joint pain": "arthralgia",
    "muscle pain": "myalgia",
    "hair loss": "alopecia",
    "headache": "cephalgia",
    "difficulty swallowing": "dysphagia",
    "high heart rate": "tachycardia",
    "fast heart rate": "tachycardia",
    "rapid heart rate": "tachycardia",
    "heart rate": "tachycardia"
}

SYNONYMS = {
    "obesity": ["obesity", "overweight", "fat", "adiposity"],
    "hyperglycemia": ["hyperglycemia", "high sugar", "diabetes", "high glucose"],
    "polyuria": ["polyuria", "frequent urination", "frequent micturition"],
    "fatigue": ["fatigue", "tired", "tiredness", "exhaustion", "lethargy"],
    "polydipsia": ["polydipsia", "thirst", "excessive thirst"],
    "hypertension": ["hypertension", "high blood pressure"],
    "hypotension": ["hypotension", "low blood pressure"],
    "nephropathy": ["nephropathy", "kidney disease", "kidney damage"],
    "neuropathy": ["neuropathy", "nerve pain", "nerve damage"],
    "retinopathy": ["retinopathy", "eye damage", "vision loss"],
    "edema": ["edema", "swelling"],
    "pyrexia": ["pyrexia", "fever", "high temperature"],
    "rhinitis": ["rhinitis", "runny nose", "congestion"],
    "myocardial infarction": ["myocardial infarction", "heart attack"],
    "cerebrovascular accident": ["cerebrovascular accident", "stroke"],
    "dyspnea": ["dyspnea", "shortness of breath", "breathing difficulty"],
    "arthralgia": ["arthralgia", "joint pain", "joint stiffness"],
    "myalgia": ["myalgia", "muscle pain", "muscle soreness"],
    "alopecia": ["alopecia", "hair loss", "baldness"],
    "cephalgia": ["cephalgia", "headache", "migraine"],
    "tachycardia": ["tachycardia", "high heart rate", "elevated heart rate", "heart rate", "fast heart rate"],
    "Type 2 Diabetes": ["Type 2 Diabetes", "diabetes mellitus", "T2D"],
    "Metabolic Syndrome": ["Metabolic Syndrome", "syndrome X"]
}


def normalize_query(query: str) -> str:
    """Map casual symptoms and terms to their medical equivalents."""
    normalized = query.lower()
    # Sort keys by length descending to replace longer phrases first (e.g. "high sugar" before "sugar")
    sorted_keys = sorted(CASUAL_TO_MEDICAL.keys(), key=len, reverse=True)
    for key in sorted_keys:
        pattern = r"\b" + re.escape(key) + r"\b"
        normalized = re.sub(pattern, CASUAL_TO_MEDICAL[key], normalized)
    return normalized


def expand_query(query: str) -> tuple[str, List[str]]:
    """Normalize query, infer diseases, and perform synonym/Boolean query expansion."""
    normalized = normalize_query(query)
    inferred_diseases = infer_diseases(normalized)
    
    expanded = normalized
    # Sort synonym keys by length descending to replace longer words first
    sorted_synonym_keys = sorted(SYNONYMS.keys(), key=len, reverse=True)
    
    for key in sorted_synonym_keys:
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, expanded, re.IGNORECASE):
            formatted_syns = [f'"{s}"' if " " in s else s for s in SYNONYMS[key]]
            syn_group = "(" + " OR ".join(formatted_syns) + ")"
            expanded = re.sub(pattern, syn_group, expanded, flags=re.IGNORECASE)
            
    # Append inferred diseases as AND groups if any were found
    if inferred_diseases:
        disease_groups = []
        for disease in inferred_diseases:
            syns = SYNONYMS.get(disease, [disease])
            formatted_syns = [f'"{s}"' if " " in s else s for s in syns]
            disease_groups.append("(" + " OR ".join(formatted_syns) + ")")
        if disease_groups:
            inferred_str = " AND ".join(disease_groups)
            if expanded.strip():
                expanded = f"({expanded}) AND {inferred_str}"
            else:
                expanded = inferred_str
                
    return expanded, inferred_diseases
