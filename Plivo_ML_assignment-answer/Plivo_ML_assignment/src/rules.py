import re
from typing import List
from rapidfuzz import process, fuzz

EMAIL_TOKEN_PATTERNS = [
    (r'\b\(?(at|@)\)?\b', '@'),
    (r'\b(dot)\b', '.'),
    (r'\s*@\s*', '@'),
    (r'\s*\.\s*', '.')
]

def collapse_spelled_letters(s: str) -> str:
    # Collapse sequences like 'g m a i l' -> 'gmail'
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        # lookahead for sequences of single letters
        if all(len(t)==1 for t in tokens[i:i+5]) and i+4 <= len(tokens):
            out.append(''.join(tokens[i:i+5]))
            i += 5
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)

def normalize_email_tokens(s: str) -> str:
    s2 = s
    s2 = collapse_spelled_letters(s2)
    for pat, rep in EMAIL_TOKEN_PATTERNS:
        s2 = re.sub(pat, rep, s2, flags=re.IGNORECASE)
    # remove spaces around @ and . inside emails
    s2 = re.sub(r'\s*([@\.])\s*', r'\1', s2)
    return s2

# Numbers: handle 'double nine', 'triple zero', 'oh' for zero
NUM_WORD = {
    'zero':'0','oh':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9'
}

def words_to_digits(seq: List[str]) -> str:
    out = []
    i = 0
    while i < len(seq):
        tok = seq[i].lower()
        if tok in ('double','triple') and i+1 < len(seq):
            times = 2 if tok=='double' else 3
            nxt = seq[i+1].lower()
            if nxt in NUM_WORD:
                out.append(NUM_WORD[nxt]*times)
                i += 2
                continue
        if tok in NUM_WORD:
            out.append(NUM_WORD[tok])
            i += 1
        else:
            # stop on first non-number word in a numeric phrase
            i += 1
    return ''.join(out)

def normalize_numbers_spoken(s: str) -> str:
    # Replace simple spoken digit sequences with digits
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        # greedy take up to 8 tokens for number
        window = tokens[i:i+8]
        wd = words_to_digits(window)
        if len(wd) >= 2:  # treat as a number
            out.append(wd)
            i += len(window)  # move past window
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)

def normalize_currency(s: str) -> str:
    # Replace 'rupees ...' preceding digits with ₹
    s = re.sub(r'\brupees\s+', '₹', s, flags=re.IGNORECASE)
    # Insert commas in Indian grouping (basic version) for 5+ digits
    def indian_group(num):
        # last 3, then every 2
        x = str(num)
        if len(x) <= 3: return x
        last3 = x[-3:]
        rest = x[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest: parts.insert(0, rest)
        return ','.join(parts + [last3])
    def repl(m):
        raw = re.sub('[^0-9]', '', m.group(0))
        if not raw: return m.group(0)
        return '₹' + indian_group(int(raw))
    s = re.sub(r'₹\s*[0-9][0-9,\.]*', repl, s)
    return s

def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 90) -> str:
    tokens = s.split()
    out = []
    for t in tokens:
        best = process.extractOne(t, names_lex, scorer=fuzz.ratio)
        if best and best[1] >= threshold:
            out.append(best[0])
        else:
            out.append(t)
    return ' '.join(out)

def generate_candidates(text: str, names_lex: List[str]) -> List[str]:
    cands = set()
    t = text
    # Base clean-ups
    t1 = normalize_email_tokens(t)
    t1 = normalize_numbers_spoken(t1)
    t1 = normalize_currency(t1)
    t1 = correct_names_with_lexicon(t1, names_lex)
    cands.add(t1)

    # Variants
    # Variant 2: only email normalization
    t2 = correct_names_with_lexicon(normalize_email_tokens(text), names_lex)
    cands.add(t2)

    # Variant 3: currency + numbers only
    t3 = normalize_currency(normalize_numbers_spoken(text))
    cands.add(t3)

    # Variant 4: names only
    t4 = correct_names_with_lexicon(text, names_lex)
    cands.add(t4)

    # ensure original too
    cands.add(text)

    # Deduplicate and limit
    out = list(cands)
    out = sorted(out, key=lambda x: len(x))[:5]  # simple cap
    return out

import re
import json
import string
from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz, process

class RuleBasedCorrector:
    def __init__(self, names_lexicon_path: str, misspell_map_path: str):
        self.names_lexicon = self.load_names_lexicon(names_lexicon_path)
        self.misspell_map = self.load_misspell_map(misspell_map_path)
        
        # Enhanced email patterns
        self.email_patterns = [
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            r'\b[a-zA-Z0-9._%+-]+\s*(?:at|where)\s*[a-zA-Z0-9.-]+\s*(?:dot|\.)\s*[a-zA-Z]{2,}\b',
            r'\b[a-zA-Z0-9._%+-]+\s*at\s*[a-zA-Z0-9.-]+\s*dot\s*(?:com|org|in|net|edu)\b'
        ]
        
        # Indian number formats
        self.indian_number_patterns = [
            r'\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b',  # Mobile numbers
            r'\b\d{2,4}[\-\s]?\d{6,8}\b',  # Landline numbers
            r'\b\d{1,2}[\-\s]?\d{3,4}[\-\s]?\d{4}\b'  # Various formats
        ]
        
        # Currency patterns
        self.currency_patterns = [
            r'(?:₹|rs|rupees?)\s*(\d+(?:,\d{2,3})*(?:\.\d{1,2})?)',
            r'(\d+(?:,\d{2,3})*(?:\.\d{1,2})?)\s*(?:₹|rs|rupees?)'
        ]

    def load_names_lexicon(self, path: str) -> List[str]:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f if line.strip()]

    def load_misspell_map(self, path: str) -> Dict[str, str]:
        with open(path, 'r') as f:
            return json.load(f)

    def generate_candidates(self, text: str) -> List[str]:
        candidates = set()
        candidates.add(text)  # Original text
        
        # Apply various correction strategies
        self.apply_misspell_correction(candidates, text)
        self.apply_email_correction(candidates, text)
        self.apply_number_correction(candidates, text)
        self.apply_currency_correction(candidates, text)
        self.apply_name_correction(candidates, text)
        self.apply_punctuation_correction(candidates, text)
        
        return list(candidates)[:20]  # Limit to 20 candidates

    def apply_misspell_correction(self, candidates: set, text: str):
        words = text.split()
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word in self.misspell_map:
                new_words = words.copy()
                new_words[i] = self.misspell_map[lower_word]
                candidates.add(' '.join(new_words))

    def apply_email_correction(self, candidates: set, text: str):
        words = text.lower().split()
        
        # Look for email-like patterns
        for i in range(len(words)):
            # Handle "at" and "dot" patterns
            if i + 2 < len(words) and words[i+1] in ['at', 'where']:
                if 'dot' in words[i+2] or any(domain in words[i+2] for domain in ['com', 'org', 'in', 'net']):
                    email = f"{words[i]}@{words[i+2].replace('dot', '.')}"
                    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                        new_text = text.replace(f"{words[i]} {words[i+1]} {words[i+2]}", email)
                        candidates.add(new_text)
            
            # Handle standalone email components
            if '@' in words[i] and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', words[i]):
                # Fix common email issues
                fixed_email = re.sub(r'[^a-zA-Z0-9@._%+-]', '', words[i])
                if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', fixed_email):
                    new_text = text.replace(words[i], fixed_email)
                    candidates.add(new_text)

    def apply_number_correction(self, candidates: set, text: str):
        words = text.split()
        new_words = words.copy()
        changed = False
        
        for i, word in enumerate(new_words):
            lower_word = word.lower()
            
            # Convert "oh" to "0"
            if lower_word == 'oh' and (i == 0 or not words[i-1].isalpha()):
                new_words[i] = '0'
                changed = True
            
            # Handle "double" pattern
            elif lower_word == 'double' and i + 1 < len(words):
                new_words[i] = words[i+1] * 2
                new_words[i+1] = ''
                changed = True
            
            # Handle Indian number formats
            elif re.match(r'^\d{10}$', word):  # Format 10-digit numbers
                formatted = f"{word[:5]} {word[5:]}"
                new_words[i] = formatted
                changed = True
        
        if changed:
            candidates.add(' '.join([w for w in new_words if w]))
        
        # Generate number variants
        self.generate_number_variants(candidates, text)

    def generate_number_variants(self, candidates: set, text: str):
        # Find and reformat numbers
        number_matches = re.findall(r'\b\d+\b', text)
        for match in number_matches:
            if len(match) == 10:  # Indian mobile number
                # Various formats
                variants = [
                    f"{match[:5]} {match[5:]}",
                    f"{match[:3]} {match[3:6]} {match[6:]}",
                    f"+91 {match}",
                    f"+91-{match}"
                ]
                for variant in variants:
                    candidates.add(text.replace(match, variant))

    def apply_currency_correction(self, candidates: set, text: str):
        # Handle ₹ symbol and formatting
        for pattern in self.currency_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                amount = match.group(1)
                # Indian number formatting
                formatted_amount = self.format_indian_number(amount)
                new_text = text.replace(amount, formatted_amount)
                candidates.add(new_text)
                
                # Add ₹ symbol if missing
                if '₹' not in text:
                    candidates.add(f"₹ {formatted_amount}")

    def format_indian_number(self, num_str: str) -> str:
        """Format numbers in Indian numbering system"""
        try:
            # Remove commas and convert to float
            num = float(num_str.replace(',', ''))
            
            # Format with Indian comma separation
            num_parts = f"{num:,.2f}".split('.')
            integer_part = num_parts[0]
            
            # Indian numbering system (lakhs, crores)
            if len(integer_part) > 3:
                last_three = integer_part[-3:]
                other_numbers = integer_part[:-3]
                if len(other_numbers) > 2:
                    formatted = f"{other_numbers[:-2]},{other_numbers[-2:]},{last_three}"
                else:
                    formatted = f"{other_numbers},{last_three}"
            else:
                formatted = integer_part
            
            return formatted
        except:
            return num_str

    def apply_name_correction(self, candidates: set, text: str):
        words = text.split()
        
        for i, word in enumerate(words):
            if len(word) > 2 and word.isalpha():
                # Fuzzy match with names lexicon
                best_match, score, _ = process.extractOne(
                    word.lower(), 
                    self.names_lexicon, 
                    scorer=fuzz.ratio
                )
                
                if score > 85:  # High confidence match
                    new_words = words.copy()
                    new_words[i] = best_match.title()
                    candidates.add(' '.join(new_words))

    def apply_punctuation_correction(self, candidates: set, text: str):
        # Add sentence-final punctuation if missing
        if text and text[-1] not in ['.', '!', '?']:
            candidates.add(text + '.')
        
        # Capitalize first letter
        if text and text[0].islower():
            candidates.add(text[0].upper() + text[1:])
        
        # Fix multiple spaces
        if '  ' in text:
            candidates.add(re.sub(r'\s+', ' ', text))

def correct_text(text: str, names_lexicon_path: str, misspell_map_path: str) -> List[str]:
    corrector = RuleBasedCorrector(names_lexicon_path, misspell_map_path)
    return corrector.generate_candidates(text)