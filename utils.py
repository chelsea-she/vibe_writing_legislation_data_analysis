# To install the spaCy library, run the following command in your terminal:
# pip install spacy

# To download the medium-sized English language model, run the following command in your terminal:
# python -m spacy download en_core_web_md

import spacy

nlp = spacy.load("en_core_web_md")

from datetime import datetime
import re
import pandas as pd
import os
from extract_coauthor_raw_logs import jsonl_names
import numpy as np


def sent_tokenize(text):
    # Normalize multiple whitespace to single spaces
    text = re.sub(r"\s+", " ", text.strip())

    # First, extract all $...$ segments and replace them with placeholders
    dollar_extracted = []

    def replace_dollar(match):
        full_match = match.group(0)  # includes the dollar signs
        dollar_extracted.append(full_match)
        return f"[[DOLLAR_{len(dollar_extracted)-1}]]"

    text = re.sub(r"\$[^$]+\$", replace_dollar, text)

    # Split on sentence-ending punctuation
    pattern = r"([.?!]+)(?=\s|$)"
    pieces = re.split(pattern, text)

    # Reconstruct sentences
    sentences = []
    for i in range(0, len(pieces), 2):
        sentence = pieces[i].strip()
        if i + 1 < len(pieces):
            sentence += pieces[i + 1]
        if sentence:
            sentences.append(sentence.strip())

    # Replace placeholders with original $...$ chunks as full sentences
    final_sentences = []
    for sentence in sentences:
        parts = re.split(r"(\[\[DOLLAR_\d+\]\])", sentence)
        for part in parts:
            match = re.match(r"\[\[DOLLAR_(\d+)\]\]", part)
            if match:
                final_sentences.append(dollar_extracted[int(match.group(1))].strip())
            elif part.strip():
                final_sentences.append(part.strip())

    return final_sentences


def find_last_punctuation(text):
    punctuation = ["!?."]
    for i in range(len(text) - 1, -1, -1):
        if text[i] in punctuation:
            return i
    return 0


def find_last_suggestion(text):
    last_dollar = False
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "$" and last_dollar:
            return i
        if text[i] == "$":
            last_dollar = True
    return 0


def extract_prompt(text):
    second = text.rfind("$")
    first = text[:second].rfind("$")
    if first != -1 and second != -1:
        return text[first + 1 : second].strip()
    return None


def shorten_tokenizer(text, tokenizer):
    tokens = tokenizer.tokenize(joined_args)

    if len(tokens) > 450:
        tokens = tokens[-450:]

    # Decode back to string
    joined_args = tokenizer.convert_tokens_to_string(tokens)


def get_spacy_similarity(text1, text2, nouns_only=False):
    if nouns_only:
        doc1 = nlp(
            " ".join([str(t) for t in nlp(text1) if t.pos_ in ["NOUN", "PROPN"]])
        )
        doc2 = nlp(
            " ".join([str(t) for t in nlp(text2) if t.pos_ in ["NOUN", "PROPN"]])
        )
    else:
        doc1 = nlp(" ".join([str(t) for t in nlp(text1) if not t.is_stop]))
        doc2 = nlp(" ".join([str(t) for t in nlp(text2) if not t.is_stop]))

    if len(doc1) == 0 or len(doc2) == 0:
        return 0

    return doc1.similarity(doc2)


def get_timestamp(timestamp):
    real_timestamp = int(timestamp / 1000)
    return datetime.fromtimestamp(real_timestamp)


def convert_timestamp_to_string(timestamp):
    return timestamp.strftime("%Y/%m/%d %H:%M:%S")


def convert_string_to_timestamp(date_string):
    return datetime.strptime(date_string, "%Y/%m/%d %H:%M:%S")


def custom_serializer(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def get_filename(folder_name, session_id):
    return "formal_raw_logs/" + folder_name + session_id + ".jsonl"


background_info = [
    """Zero-price markets," where companies offer goods or services for free, have rapidly 
  increased in number and variety, including areas like software, social media, and travel booking. 
  However, antitrust regulators have struggled to address these markets because traditional antitrust 
  laws are based on the idea that prices are essential for market analysis. This focus on pricing has 
  led to significant consumer harm being overlooked, as seen in the deregulation of the radio industry, 
  where potential negative impacts on listeners were ignored during merger reviews. 
  Some legal philosophies argue that zero-price markets should still be considered in antitrust discussions, as they can involve costs 
  that signal market behavior. Ignoring these markets could lead to unfair and inefficient outcomes, 
  so antitrust laws should adapt to address them effectively. Others argue that since zero-price 
  markets often rely on alternative revenue models, such as advertising, they do not pose the same risks as 
  traditional markets, suggesting that consumer welfare is maintained through these different mechanisms.""",
    """Corporate personhood, also known as juridical personality, is the legal concept that allows corporations to
  be treated as legal entities separate from their owners, managers, and employees. This means that corporations, like 
  individuals, can own property, enter contracts, sue or be sued, and have certain legal rights and responsibilities
  In Citizens United v. FEC, the Court majority interpreted campaign spending as an exercise of free speech, more specifically 
  political speech, and granted corporations protection from government restrictions on campaign spending. Critics have blamed 
  corporate personhood for expanding corporate rights and political influence, particularly in campaign financing. However, 
  proponents argue that courts generally extend rights to corporations not to benefit them as entities, but to protect the 
  constitutional rights of their human stakeholders—such as owners and shareholders. Instead of blaming corporate personhood, 
  proponents believe corporate empowerment might stem more from economic factors such as limited regulation, corporate longevity, and globalization.""",
]
