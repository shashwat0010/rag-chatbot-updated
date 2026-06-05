import re
from typing import List, Set

SYMPTOM_DISEASE_MAPPING = [
    {
        "symptoms": {"polyuria", "polydipsia", "obesity"},
        "disease": "Type 2 Diabetes"
    },
    {
        "symptoms": {"fatigue", "polydipsia", "polyuria"},
        "disease": "Type 2 Diabetes"
    },
    {
        "symptoms": {"fatigue", "polydipsia", "polyuria", "hyperglycemia"},
        "disease": "Type 2 Diabetes"
    },
    {
        "symptoms": {"chest pain", "dyspnea"},
        "disease": "Myocardial Infarction"
    },
    {
        "symptoms": {"arthralgia", "edema"},
        "disease": "Rheumatoid Arthritis"
    },
    {
        "symptoms": {"hypertension", "cephalgia"},
        "disease": "Preeclampsia"
    },
    {
        "symptoms": {"pyrexia", "cough", "dyspnea"},
        "disease": "Pneumonia"
    },
    {
        "symptoms": {"obesity", "hyperglycemia", "hypertension"},
        "disease": "Metabolic Syndrome"
    }
]

SYMPTOMS_LIST = {
    "fatigue", "polydipsia", "polyuria", "hyperglycemia", "obesity", 
    "hypertension", "hypotension", "nephropathy", "neuropathy", 
    "retinopathy", "edema", "pyrexia", "rhinitis", "myocardial infarction", 
    "cerebrovascular accident", "dyspnea", "arthralgia", "myalgia", 
    "alopecia", "cephalgia", "dysphagia", "cough", "tachycardia"
}


def infer_diseases(normalized_query: str) -> List[str]:
    """Detect symptoms in the normalized query and infer corresponding diseases."""
    detected_symptoms: Set[str] = set()
    normalized_lower = normalized_query.lower()
    
    for symptom in SYMPTOMS_LIST:
        # Match as word boundary to prevent partial word matches
        pattern = r"\b" + re.escape(symptom) + r"\b"
        if re.search(pattern, normalized_lower):
            detected_symptoms.add(symptom)
            
    inferred: List[str] = []
    for mapping in SYMPTOM_DISEASE_MAPPING:
        if mapping["symptoms"].issubset(detected_symptoms):
            if mapping["disease"] not in inferred:
                inferred.append(mapping["disease"])
                
    return inferred
