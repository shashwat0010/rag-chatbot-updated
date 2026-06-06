import re
from typing import Tuple, List

def is_gibberish(word: str) -> bool:
    """Check if a word is likely gibberish."""
    # Check for excessive consecutive consonants (5 or more)
    if re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', word, re.IGNORECASE):
        return True
    
    # Check for known keyboard mash patterns
    mash_patterns = [r'asdf', r'qwerty', r'zxcv', r'hjkl']
    for p in mash_patterns:
        if re.search(p, word, re.IGNORECASE):
            return True
            
    # Check for no vowels in a word longer than 3 characters that contains letters
    if len(word) > 3 and re.search(r'[a-zA-Z]', word) and not re.search(r'[aeiouy]', word, re.IGNORECASE):
        return True
        
    return False

def assess_query_quality(query: str) -> Tuple[bool, str, float, List[str]]:
    """
    Assess the quality of the query.
    Returns: (is_valid, reason, quality_score, ignored_tokens)
    """
    words = query.split()
    if not words:
        return False, "Query is empty.", 0.0, []
        
    ignored_tokens = []
    valid_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word)
        if not clean_word:
            continue
        if is_gibberish(clean_word):
            ignored_tokens.append(word)
        else:
            valid_words.append(word)
            
    quality_score = len(valid_words) / len(words) if words else 0.0
    
    # Require at least some substantive valid words
    if len(valid_words) < 2:
        return False, "The query appears incomplete or contains unsupported terms. Please rephrase using clearer medical terminology.", quality_score, ignored_tokens
        
    if quality_score <= 0.5:
        return False, "The query contains too many unrecognized terms. Please rephrase using clearer medical terminology.", quality_score, ignored_tokens
        
    return True, "", quality_score, ignored_tokens
