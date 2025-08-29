import re
from typing import Tuple, Dict, Any

def correct_mrz_name_filler(raw_line1: str) -> Tuple[str, str, str]:
    """
    Fix likely OCR mistakes in MRZ line1 name field where '<' was read as 'K'.
    Input: raw_line1 (string, can be shorter or longer than 44 chars)
    Output: (corrected_line1, surname, given_names)
    """
    # normalize input and ensure at least 44 chars to simplify slices
    if not isinstance(raw_line1, str):
        raw_line1 = "" if raw_line1 is None else str(raw_line1)
    line = raw_line1.upper()
    # ensure length >= 44 by padding with '<' (MRZ filler)
    if len(line) < 44:
        line = line.ljust(44, '<')
    else:
        # if longer, keep start 44 chars (we assume first 44 are MRZ)
        line = line[:44]

    # name portion is positions 5..43 (inclusive)
    name_portion = list(line[5:44])  # list of chars for mutability
    L = len(name_portion)

    # helper predicates
    def is_letter(ch: str) -> bool:
        return bool(re.match(r'^[A-Z]$', ch))

    def is_filler_like(ch: str) -> bool:
        # we treat explicit '<' as filler; also any char that is not A-Z is filler-like
        return ch == '<' or not is_letter(ch)

    # Step 1: convert obviously invalid chars to '<' (but keep A-Z and '<')
    for i, ch in enumerate(name_portion):
        if not is_letter(ch) and ch != '<':
            # convert digits and other symbols to filler
            name_portion[i] = '<'

    # Step 2: find filler runs (consecutive positions where is_filler_like True)
    filler_runs = []
    i = 0
    while i < L:
        if is_filler_like(name_portion[i]):
            j = i
            while j < L and is_filler_like(name_portion[j]):
                j += 1
            filler_runs.append((i, j))  # start inclusive, end exclusive
            i = j
        else:
            i += 1

    # Step 3: Correct 'K' occurrences that are likely filler:
    # - If a 'K' appears inside a recognized filler run -> replace with '<'
    # - If trailing region (after majority of letters) has runs of 'K' or 'K' mixed with '<', convert them
    # Compute letter_count and filler density
    letter_count = sum(1 for ch in name_portion if is_letter(ch))
    filler_count = L - letter_count
    filler_density = filler_count / max(1, L)

    # Replace 'K' characters inside filler_runs
    for start, end in filler_runs:
        for pos in range(start, end):
            if name_portion[pos] == 'K':
                name_portion[pos] = '<'

    # If filler density is high (>= 0.25) or trailing characters are mostly K/'<' -> convert trailing Ks
    trailing = ''.join(name_portion).rstrip('<')
    trailing_idx = len(trailing)  # index of first trailing filler if any
    trailing_segment = name_portion[trailing_idx:] if trailing_idx < L else []
    if filler_density >= 0.25:
        # convert Ks in trailing segment
        for idx in range(trailing_idx, L):
            if name_portion[idx] == 'K':
                name_portion[idx] = '<'

    # Extra heuristic: near existing explicit '<' neighbors, K is almost certainly a misread
    for idx in range(L):
        ch = name_portion[idx]
        if ch == 'K':
            left = name_portion[idx-1] if idx-1 >= 0 else None
            right = name_portion[idx+1] if idx+1 < L else None
            if (left == '<') or (right == '<'):
                name_portion[idx] = '<'

    # Build corrected name portion and full corrected line1
    corrected_name_portion = ''.join(name_portion)
    corrected_line1 = line[:5] + corrected_name_portion  # keep full length 44

    # Now parse surname and given names by splitting on '<<'
    if '<<' in corrected_name_portion:
        parts = corrected_name_portion.split('<<', 1)
        surname_raw = parts[0].replace('<', ' ').strip()
        given_raw = parts[1].replace('<', ' ').strip()
    else:
        # If no clear '<<', attempt to find first run of 2+ fillers and split there; else fallback
        split_idx = None
        for (s, e) in filler_runs:
            if e - s >= 2:
                split_idx = s
                break
        if split_idx is None:
            # fallback: first single '<' after at least 2 letters
            letter_run = 0
            for idx, ch in enumerate(name_portion):
                if is_letter(ch):
                    letter_run += 1
                elif letter_run >= 2:
                    split_idx = idx
                    break
                else:
                    letter_run = 0
        if split_idx is None:
            # no separator found -> put all as surname
            surname_raw = ''.join(name_portion).replace('<', ' ').strip()
            given_raw = ''
        else:
            surname_raw = ''.join(name_portion[:split_idx]).replace('<', ' ').strip()
            given_raw = ''.join(name_portion[split_idx:]).replace('<', ' ').strip()

    # Final tidy: collapse multiple spaces
    surname = re.sub(r'\s+', ' ', surname_raw).strip()
    given_names = re.sub(r'\s+', ' ', given_raw).strip()

    return corrected_line1, surname, given_names


def apply_name_corrections_to_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a parsed dict that contains raw_mrz_line1 (or raw_line1), apply corrections to the name field
    and return a copy of parsed with corrected values:
      - updates 'raw_mrz_line1'
      - sets/overwrites 'surname' and 'given_names'
    """
    p = dict(parsed)  # shallow copy
    raw_line1 = p.get('raw_mrz_line1') or p.get('raw_line1') or ""
    corrected_line1, surname, given_names = correct_mrz_name_filler(raw_line1)
    p['raw_mrz_line1'] = corrected_line1
    p['surname'] = surname or p.get('surname')  # prefer corrected if non-empty
    p['given_names'] = given_names or p.get('given_names')
    return p
