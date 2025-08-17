import utils

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

from parascore import ParaScorer

import spacy
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher

from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import re

import pandas as pd
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"

stance = pipeline("zero-shot-classification", model="roberta-large-mnli")
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")


def get_stance_difference(past_args, prompt):
    joined_args = past_args

    hypothesis_template = f"This text {{}} with the view that: {joined_args}"

    result = stance(
        prompt,
        candidate_labels=["agrees", "disagrees", "neutral"],
        hypothesis_template=hypothesis_template,
    )
    return result


# Load pre-trained NLI model
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(
    device
)


def get_NLI(past_args, prompt):
    joined_args = past_args
    # print(nli_tokenizer.tokenize(joined_args))

    inputs = nli_tokenizer.encode_plus(
        joined_args,
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    try:
        outputs = nli_model(**inputs)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e

    probs = F.softmax(outputs.logits, dim=1)

    # Map labels
    labels = ["entailment", "neutral", "contradiction"]
    predicted = labels[probs.argmax()]

    # Show results
    labelDict = {}
    for label, prob in zip(labels, probs[0]):
        labelDict[label] = f"{prob.item():.4f}"
        # print(f"{label}: {prob.item():.4f}")

    # print(f"\nPredicted relationship: {predicted}")
    return predicted, labelDict


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_constructive_learning(prewriting, writing):
    # Step 1: Split into sentences
    sentences_a = utils.sent_tokenize(prewriting)

    # Step 2: Embed all sentences
    embeddings_a = embedding_model.encode(sentences_a, convert_to_tensor=True)
    embeddings_b = embedding_model.encode(writing, convert_to_tensor=True)

    # Step 3: Compare every sentence from A with every sentence from B
    similarities = util.pytorch_cos_sim(embeddings_a, embeddings_b)

    # Step 4: Flag matches
    threshold = 0.75
    repeats = []

    for i, row in enumerate(similarities):
        for j, score in enumerate(row):
            if score > threshold:
                repeats.append((sentences_a[i], writing[j], float(score)))

    # Output repeated ideas
    repeat_dict = {}
    for a, b, score in repeats:
        if b in repeat_dict:
            repeat_dict[b]["count"] += 1
            repeat_dict[b]["scores"].append(score)
            repeat_dict[b]["prewriting_sentences"].append(a)
        else:
            repeat_dict[b] = {
                "count": 1,
                "scores": [score],
                "prewriting_sentences": [a],
            }
            # print(f"\nText A: {a}\nText B: {b}\nSimilarity: {score:.2f}")
    return repeat_dict


def get_relevant_background_info(session_id, writing):
    # Step 1: Split into sentences
    background = (
        utils.background_info[0]
        if "antitrust" in session_id
        else utils.background_info[1]
    )
    sentences_a = utils.sent_tokenize(background)

    # Step 2: Embed all sentences
    embeddings_a = embedding_model.encode(sentences_a, convert_to_tensor=True)
    embeddings_b = embedding_model.encode(writing, convert_to_tensor=True)

    # Step 3: Compare every sentence from A with every sentence from B
    similarities = util.pytorch_cos_sim(embeddings_a, embeddings_b)

    # Step 4: Flag matches
    threshold = 0.5
    repeats = []

    for i, row in enumerate(similarities):
        for j, score in enumerate(row):
            if score > threshold:
                repeats.append((sentences_a[i], writing[j], float(score)))

    # Output repeated ideas
    repeat_dict = {}
    for a, b, score in repeats:
        if b in repeat_dict:
            repeat_dict[b]["count"] += 1
            repeat_dict[b]["scores"].append(score)
            repeat_dict[b]["prewriting_sentences"].append(a)
        else:
            repeat_dict[b] = {
                "count": 1,
                "scores": [score],
                "prewriting_sentences": [a],
            }
            # print(f"\nText A: {a}\nText B: {b}\nSimilarity: {score:.2f}")
    return repeat_dict


def parse_level_2_constructive_learning(repeat_dict, writing):
    # writing_sentences = utils.sent_tokenize(writing)
    num_sentences = len(writing)
    return round(len(repeat_dict) / num_sentences, 2)


scorer = ParaScorer(lang="en", model_type="bert-base-uncased")


def get_paraphrase_depth(writing, repeat_dict):
    if not repeat_dict:
        return 0

    parascore_dict = {}

    for writing_sentence, info in repeat_dict.items():
        for prewriting_sentence in info["prewriting_sentences"]:
            # Both inputs must be lists
            score = float(scorer.score([prewriting_sentence], [writing_sentence])[0])

            if writing_sentence in parascore_dict:
                parascore_dict[writing_sentence]["scores"].append(score)
                parascore_dict[writing_sentence]["score_sum"] += score
            else:
                parascore_dict[writing_sentence] = {
                    "scores": [score],
                    "score_sum": score,
                }

    # Average scores for each writing sentence
    for sentence_data in parascore_dict.values():
        sentence_data["score_avg"] = round(
            sentence_data["score_sum"] / len(sentence_data["scores"]), 2
        )

    return parascore_dict


def parse_level_2_paraphrase_depth(parascore_dict):
    if not parascore_dict:
        return 0

    total_score = sum(data["score_avg"] for data in parascore_dict.values())
    return round(total_score / len(parascore_dict), 2)


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Trigger phrases that often signal external examples
trigger_phrases = [
    "as part of that",
    "first",
    "for example",
    "for instance",
    "for one",
    "in fact",
    "in particular",
    "in this case",
    "indeed",
    "specifically",
]


# Build trigger matcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(text) for text in trigger_phrases]
matcher.add("TriggerPhrase", patterns)


def get_external_example(essay):
    results_dict = {}

    # Analyze each sentence
    for sent in essay:
        sent_doc = nlp(sent)

        # NER entities
        entities = [ent.label_ for ent in sent_doc.ents]
        entity_types = set(entities)

        # Count proper nouns
        proper_nouns = [token.text for token in sent_doc if token.pos_ == "PROPN"]
        proper_noun_count = len(proper_nouns)

        # Check trigger phrases
        matches = matcher(sent_doc)
        has_trigger = len(matches) > 0

        # Decision rules
        if proper_noun_count >= 1 or has_trigger:
            external = True
        else:
            external = False

        results_dict[sent.strip()] = {
            "entities": entities,
            "proper_nouns": proper_nouns,
            "has_trigger": has_trigger,
            "external_example": external,
        }
    return results_dict


def parse_level_2_external_examples(external_dict):
    if not external_dict:
        return 0

    # Count sentences with external examples
    external_count = sum(
        1 for data in external_dict.values() if data["external_example"]
    )

    # Total sentences
    total_sentences = len(external_dict)

    # Calculate percentage
    if total_sentences == 0:
        return 0.0

    return round(external_count / total_sentences, 2)


def get_relevant_past_suggestions(past_suggestions, writing):
    if not past_suggestions or not writing:
        return {}
    # Step 2: Embed all sentences
    embeddings_a = embedding_model.encode(past_suggestions, convert_to_tensor=True)
    embeddings_b = embedding_model.encode(writing, convert_to_tensor=True)

    # Step 3: Compare every sentence from A with every sentence from B
    similarities = util.pytorch_cos_sim(embeddings_a, embeddings_b)

    # Step 4: Flag matches
    threshold = 0.5
    repeats = []

    for i, row in enumerate(similarities):
        for j, score in enumerate(row):
            if score > threshold:
                repeats.append((past_suggestions[i], writing[j], float(score)))

    # Output repeated ideas
    repeat_dict = {}
    for a, b, score in repeats:
        if b in repeat_dict:
            repeat_dict[b]["count"] += 1
            repeat_dict[b]["scores"].append(score)
            repeat_dict[b]["ai_suggestions"].append(a)
        else:
            repeat_dict[b] = {
                "count": 1,
                "scores": [score],
                "ai_suggestions": [a],
            }
            # print(f"\nText A: {a}\nText B: {b}\nSimilarity: {score:.2f}")
    return repeat_dict


def get_paraphrase_depth_suggestions(repeat_dict):
    if not repeat_dict:
        return 0

    parascore_dict = {}

    for writing_sentence, info in repeat_dict.items():
        for suggestion in info["ai_suggestions"]:
            # Both inputs must be lists
            score = float(scorer.score([suggestion], [writing_sentence])[0])

            if writing_sentence in parascore_dict:
                parascore_dict[writing_sentence]["scores"].append(score)
                parascore_dict[writing_sentence]["score_sum"] += score
            else:
                parascore_dict[writing_sentence] = {
                    "scores": [score],
                    "score_sum": score,
                }

    # Average scores for each writing sentence
    for sentence_data in parascore_dict.values():
        sentence_data["score_avg"] = round(
            sentence_data["score_sum"] / len(sentence_data["scores"]), 2
        )

    return parascore_dict


classifier_text = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def classify_text(text):
    result = classifier_text(
        text,
        candidate_labels=["question", "imperative", "statement"],
    )
    return dict(zip(result["labels"], result["scores"]))


def is_imperative(text):
    doc = nlp(text.strip())
    if not doc:
        return False

    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in {"ROOT"}:
            if token.tag_ == "VB":
                return True
        elif token.pos_ in {"NOUN", "PRON"}:
            break

    return False


def syntactic_heuristic(text):
    text = text.strip()
    text_lower = text.lower()
    if text_lower.endswith("?") or re.match(
        r"(?i)^(who|what|when|where|why|how|is|are|can|do|does|could|would)\b", text
    ):
        return "question"
    elif is_imperative(text):
        return "imperative"
    else:
        return "statement"


def parse_classify_text(text):
    zero_shot_result = classify_text(text)
    heur_label = syntactic_heuristic(text)

    sorted_scores = sorted(zero_shot_result.values(), reverse=True)
    top_label, top_score = max(zero_shot_result.items(), key=lambda x: x[1])

    if top_score >= 0.75 and top_label == heur_label:
        return {top_label: top_score}
    elif top_score < 0.65 or (top_score - sorted_scores[1]) < 0.15:
        return {heur_label: zero_shot_result.get(heur_label, 0.0)}
    else:
        return {heur_label: zero_shot_result.get(heur_label, 0.0)}


#### CHECK FOR MULTIPLE PERSPECTIVES ####
connective_categories = {
    "Comparison": [
        "although",
        "as though",
        "but",
        "by comparison",
        "by contrast",
        "conversely",
        "despite",
        "furthermore",
        "granted",
        "however",
        "if",
        "in contrast",
        "in fact",
        "indeed",
        "much as",
        "nevertheless",
        "nonetheless",
        "nor",
        "on the contrary",
        "on the other hand",
        "rather",
        "regardless",
        "still",
        "then",
        "though",
        "while",
        "whereas",
        "yet",
    ],
    "Expansion": [
        "or",
        "alternatively",
        "otherwise",
        "instead",
        "either",
        "else",
        "except",
        "separately",
    ],
}

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
connective_to_category = {}

for category, phrases in connective_categories.items():
    category_str = str(category)
    patterns = [nlp.make_doc(phrase) for phrase in phrases]
    matcher.add(category_str, patterns)  # Add matcher
    _ = nlp.vocab.strings[
        category_str
    ]  # Register *after* matcher to ensure consistency
    for phrase in phrases:
        connective_to_category[phrase.lower()] = category_str


def get_discourse_connectives(text_sents):
    results = []

    for sent in text_sents:
        sent_doc = nlp(sent)
        matches = matcher(sent_doc)

        for match_id, start, end in matches:
            span = sent_doc[start:end]
            matched_text = span.text.lower()
            try:
                category = nlp.vocab.strings[match_id]
            except KeyError:
                category = "UNKNOWN"

            results.append(
                {
                    "connective": matched_text,
                    "category": category,
                    "sentence": sent,
                }
            )

    matches_by_sent = {}
    for result in results:
        if result["sentence"] in matches_by_sent:
            matches_by_sent[result["sentence"]]["categories"].append(result["category"])
            matches_by_sent[result["sentence"]]["connectives"].append(
                result["connective"]
            )
        else:
            matches_by_sent[result["sentence"]] = {
                "categories": [result["category"]],
                "connectives": [result["connective"]],
            }

    return matches_by_sent


classifier_arg = pipeline("text-classification", model="chkla/roberta-argument")


def classify_argument_sentences(paragraph):
    sentences = utils.sent_tokenize(paragraph)
    results = []
    for sent in sentences:
        pred = classifier_arg(sent)[0]
        results.append(
            {
                "sentence": sent,
                "is_argument": pred["label"] == "ARGUMENT",
                "score": pred["score"],
            }
        )
        # print(f"'{sent}' → {pred}")
    return [r for r in results if r["is_argument"]]


model = SentenceTransformer("all-MiniLM-L6-v2")


def parse_argument_sentences(arguments, all_distinct_arguments):
    num_new_args = 0
    new_args = []

    for argInfo in arguments:
        i = 0
        isNew = True
        while i < len(all_distinct_arguments):
            emb1 = model.encode(all_distinct_arguments[i], convert_to_tensor=True)
            emb2 = model.encode(argInfo["sentence"], convert_to_tensor=True)

            similarity = util.pytorch_cos_sim(emb1, emb2).item()
            if similarity >= 0.75:
                isNew = False
                break
            i += 1
        if isNew:
            num_new_args += 1
            all_distinct_arguments.append(argInfo["sentence"])
            new_args.append(argInfo["sentence"])

    return {"new_arguments": new_args}


#### CHECK FOR METACOGNITION ####
metacognitive_verbs = [
    "assume",
    "agree",
    "attempt",
    "believe",
    "care",
    "confuse",
    "decide",
    "determine",
    "digest",
    "discover",
    "doubt",
    "enjoy",
    "expect",
    "feel",
    "figure",
    "find",
    "forget",
    "guess",
    "hope",
    "ignore",
    "imagine",
    "know",
    "learn",
    "like",
    "love",
    "mistook",
    "notice",
    "plan",
    "pretend",
    "realize",
    "reflect",
    "refuse",
    "remember",
    "see",
    "supress",
    "think",
    "understand",
    "want",
    "wonder",
]
first_person_pronouns = ["i", "we", "my", "me", "our", "us", "myself", "ourselves"]


def detect_first_person_metacognitive_phrases(insert_sents):
    matched_phrases = []

    for sent in insert_sents:
        sent_doc = nlp(sent)
        for token in sent_doc:
            # Check if token is a metacognitive verb
            if token.lemma_ in metacognitive_verbs:
                # Check for a first-person subject in its syntactic tree
                for child in token.children:
                    if (
                        child.dep_ in ("nsubj", "nsubjpass")
                        and child.text.lower() in first_person_pronouns
                    ):
                        matched_phrases.append({sent: child.text + " " + token.text})
                        break
    return matched_phrases


#### CHECK FOR COGNITIVE & EMOTIONAL ENGAGEMENT ####
hedges = set(
    {  ##VERBS
        "suggest",
        "believe",
        "appear",
        "indicate",
        "assume",
        "seem",
        "consider",
        "doubt",
        "estimate",
        "expect",
        "feel",
        "guess",
        "imagine",
        "speculate",
        "suppose",
        "think",
        "understand",
        "imply",
        "presume",
        "suspect",
        "postulate",
        "reckon",
        "infer",
        "hope",
        ##ADVERBS
        "rather",
        "slightly",
        "barely",
        "strictly",
        "presumably",
        "fairly",
        "theoretically",
        "basically",
        "relatively",
        "possibly",
        "preferably",
        "slenderly",
        "scantily",
        "decidedly",
        "arguably",
        "seemingly",
        "occasionally",
        "partially",
        "partly",
        "practically",
        "roughly",
        "virtually",
        "allegedly",
        ##ADJECTIVES
        "presumable",
        "possible",
        "probably",
        "likely",
        "apparent",
        "probable",
        "improbable",
        "unlikely",
        "rarely",
        "improbably",
        "unclearly",
        "unsure",
        "sure",
        "chance",
        "unclear",
        ##VERBS
        "may",
        "might",
        "maybe",
        "shall",
        "should",
        "can",
        "could",
        "would",
        "ought",
    }
)

boosters = set(
    {
        "clearly",
        "obviously",
        "certainly",
        "fact that",
        "show",
        "actually",
        "must",
        "of course",
        "absolutely",
        "always",
        "apparently",
        "assuredly",
        "categorically",
        "compelling",
        "completely",
        "comprehensively",
        "conclude that",
        "conclusively",
        "confirmed",
        "confirmation",
        "considerabley",
        "consistently",
        "conspicuously",
        "constantly",
        "convincingly",
        "corroboratetion",
        "crediblely",
        "crucially",
        "decisively",
        "definitely",
        "definitively",
        "demonstrate",
        "deservedly",
        "distinctively",
        "doubtlessly",
        "enhanced",
        "entirely",
        "especially",
        "essentially",
        "establish",
        "evidently",
        "exceptionally",
        "exhaustively",
        "extensively",
        "extraordinary",
        "extremely",
        "the fact that",
        "find that",
        "found that",
        "firmly",
        "forcefully",
        "fully",
        "strikingly",
        "successfully",
        "fundamentally",
        "genuinely",
        "great",
        "highlight",
        "highly",
        "impossible",
        "impressively",
        "incontrovertible",
        "indispensablely",
        "inevitabley",
        "in fact",
        "manifestly",
        "markedly",
        "meaningfully",
        "necessarily",
        "never",
        "notabley",
        "noteworthy",
        "noticeabley",
        "outstanding",
        "particularly",
        "perfectly",
        "persuasively",
        "plainly",
        "powerful",
        "precisely",
        "profoundly",
        "prominently",
        "proof",
        "proved",
        "quite",
        "radically",
        "really",
        "reliably",
        "remarkablely",
        "rigorously",
        "safely",
        "securely",
        "self-evident",
        "sizablely",
        "superior",
        "surely",
        "thoroughly",
        "totally",
        "truly",
        "unambiguously",
        "unarguably",
        "unavoidabley",
        "undeniabley",
        "undoubtedly",
        "unequivocally",
        "uniquely",
        "unlimited",
        "unmistakablely",
        "unprecedented",
        "unquestionably",
        "uphold",
        "upheld",
        "vastly",
        "vitally",
        "we know",
        "well-known",
        "indeed",
        "no doubt",
        "prove",
        "honestly",
        "mostly",
        "largely",
        "sure",
        "like i said",
        "as i say",
        "nonetheless",
        "mainly",
        "nevertheless",
    }
)


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def detect_hedging_labeled(text_sents, threshold=0.5):
    matches_by_sent = {}
    for sent in text_sents:
        sent_tokens = set(re.findall(r"\b\w+\b", sent.lower()))
        hedge_types = []

        # Check hedge words
        for word in hedges:
            if word in sent_tokens:
                hedge_types.append({"type": "hedge", "trigger_word": word})

        matches_by_sent[sent] = {"hedge_types": hedge_types}

    for sent in matches_by_sent.keys():
        matches_by_sent[sent]["hedges_count"] = len(
            matches_by_sent[sent]["hedge_types"]
        )
    return matches_by_sent


def detect_certainty(text_sents):
    matches_by_sent = {}
    for sent in text_sents:
        booster_hits = []
        for booster in boosters:
            if booster in sent.lower():
                booster_hits.append(booster)
        matches_by_sent[sent] = {
            "certainty_boosters": booster_hits,
            "booster_count": len(booster_hits),
        }
    return matches_by_sent


###SEMANTIC DIFFERENCE FUNCTIONS###
MIN_INSERT_WORD_COUNT = 10
MAJOR_INSRT_MAX_SIMILARITY = 0.9
MINOR_INSRT_MAX_SIMILARITY = 0.95
MAX_SIMILARITY = 0.95


def get_similarity_with_prev_writing_for_level_2(action, prev_writing, similarity_fcn):
    prev_sents = utils.sent_tokenize(prev_writing)
    curr_sents = utils.sent_tokenize(action["action_end_writing"])
    select_sents_after_action = action["action_modified_sentences"]
    similarity = abs(
        similarity_fcn(" ".join(select_sents_after_action), " ".join(prev_sents))
    )
    return similarity, {
        "select_sents_before_action": prev_sents,
        "select_sents_after_action": select_sents_after_action,
    }


def parse_level_2_major_insert_major_semantic_diff(
    action,
    prev_writing_similarity,
    MIN_INSERT_WORD_COUNT=MIN_INSERT_WORD_COUNT,
    MAJOR_INSRT_MAX_SIMILARITY=MAJOR_INSRT_MAX_SIMILARITY,
):
    if (
        action["action_type"] == "insert_text_human"
        and action["action_delta"] != ""
        and action["action_delta"][-1] >= MIN_INSERT_WORD_COUNT
    ):
        return prev_writing_similarity <= MAJOR_INSRT_MAX_SIMILARITY
    return False


def parse_level_2_major_insert_minor_semantic_diff(
    action,
    prev_writing_similarity,
    MIN_INSERT_WORD_COUNT=MIN_INSERT_WORD_COUNT,
    MAJOR_INSRT_MAX_SIMILARITY=MAJOR_INSRT_MAX_SIMILARITY,
):
    if (
        action["action_type"] == "insert_text_human"
        and action["action_delta"] != ""
        and action["action_delta"][-1] >= MIN_INSERT_WORD_COUNT
    ):
        return prev_writing_similarity > MAJOR_INSRT_MAX_SIMILARITY
    return False


def parse_level_2_minor_insert_major_semantic_diff(
    action,
    prev_writing_similarity,
    MIN_INSERT_WORD_COUNT=MIN_INSERT_WORD_COUNT,
    MINOR_INSRT_MAX_SIMILARITY=MINOR_INSRT_MAX_SIMILARITY,
):
    if (
        action["action_type"] == "insert_text_human"
        and action["action_delta"] != ""
        and action["action_delta"][-1] < MIN_INSERT_WORD_COUNT
    ):
        return prev_writing_similarity <= MINOR_INSRT_MAX_SIMILARITY
    return False


def parse_level_2_minor_insert_minor_semantic_diff(
    action,
    prev_writing_similarity,
    MIN_INSERT_WORD_COUNT=MIN_INSERT_WORD_COUNT,
    MINOR_INSRT_MAX_SIMILARITY=MINOR_INSRT_MAX_SIMILARITY,
):
    if (
        action["action_type"] == "insert_text_human"
        and action["action_delta"] != ""
        and action["action_delta"][-1] < MIN_INSERT_WORD_COUNT
    ):
        return prev_writing_similarity > MINOR_INSRT_MAX_SIMILARITY
    return False


def parse_level_2_delete_major_semantic_diff(
    action, prev_writing_similarity, MAX_SIMILARITY=MAX_SIMILARITY
):
    if action["action_type"] == "delete_text" and action["action_delta"] != "":
        return prev_writing_similarity <= MAX_SIMILARITY
    return False


def parse_level_2_delete_minor_semantic_diff(
    action, prev_writing_similarity, MAX_SIMILARITY=MAX_SIMILARITY
):
    if action["action_type"] == "delete_text" and action["action_delta"] != "":
        return prev_writing_similarity > MAX_SIMILARITY
    return False


model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def prepare_paragraph_A(A_sents):
    A_emb = model.encode(
        A_sents,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return A_emb


def compute_novelty_paragraphB_vs_A(
    A_sents: list,
    A_emb: np.ndarray,
    B_sents: list,
) -> pd.DataFrame:
    """
    For each sentence s_k in paragraph B, compute:
      - max_sim_to_history = max cosine similarity to any sentence in A ∪ B_{<k}
      - novelty = 1 - max_sim_to_history
      - best_match text and which paragraph it came from ("A" or "B")
    Returns a pandas DataFrame with one row per B sentence (in order).

    References:
      • Online streaming novelty via max-sim to history (ACL/HLT formulation).
      • SBERT sentence embeddings for cosine similarity (EMNLP 2019).
    """

    pool_emb = A_emb.copy()
    pool_src = ["A"] * len(A_sents)
    pool_txt = A_sents.copy()

    rows = []
    for k, s in enumerate(B_sents, start=1):
        v = model.encode(
            [s],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # cosine since embeddings are normalized
        sims = pool_emb @ v  # (N,)
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])

        novelty = 1.0 - max_sim

        rows.append(
            {
                "k": k,
                "sentence_B": s,
                "max_sim_to_history": max_sim,
                "novelty": novelty,
                "best_match_from": None if best_idx < 0 else pool_src[best_idx],
                "best_match_text": None if best_idx < 0 else pool_txt[best_idx],
            }
        )

        pool_emb = np.vstack([pool_emb, v])
        pool_src.append("B")
        pool_txt.append(s)

    return pd.DataFrame(rows)
