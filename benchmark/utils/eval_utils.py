# Copyright 2025 [Visual Spatial Tuning] Authors
import re


def extract_single_choice_with_word_boundary(pred):
    if pred is None:
        return None

    try:
        pred = str(pred)
    except Exception:
        return None

    pattern_1 = r'``([^`]*)``'
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1)

    pattern_2 = r'`([^`]*)`'
    match = re.search(pattern_2, pred)
    if match:
        pred = match.group(1)

    pattern_3 = r'\b[A-F]\b(?!\s[a-zA-Z])'
    match = re.search(pattern_3, pred)
    if match:
        pred = match.group()
    else:
        return None

    return pred
