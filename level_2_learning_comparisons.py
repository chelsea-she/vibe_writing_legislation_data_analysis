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

device = "mps" if torch.backends.mps.is_available() else "cpu"

stance = pipeline("zero-shot-classification", model="roberta-large-mnli")


def get_stance_difference(pre_writing, writing):
    trunc_pre = pre_writing
    if len(pre_writing) > 490:
        trunc_pre = pre_writing[:490]
    result = stance(
        writing,
        candidate_labels=["agrees", "disagrees", "neutral"],
        hypothesis_template="This text {} with the view that: " + trunc_pre,
    )
    return result


# Load pre-trained NLI model
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(
    device
)


def get_NLI(pre_writing, writing):
    trunc_pre = pre_writing
    if len(pre_writing) > 490:
        trunc_pre = pre_writing[:490]
        inputs = nli_tokenizer.encode_plus(
            trunc_pre, writing, return_tensors="pt", truncation=True, max_length=512
        )
    else:
        inputs = nli_tokenizer.encode_plus(
            pre_writing, writing, return_tensors="pt", truncation=True, max_length=512
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


def parse_level_2_vibe_writing(stance, nli):
    if stance == "disagrees" and nli == "contradiction":
        return 1
    elif stance == "disagrees":
        return 0.5
    elif nli == "contradiction":
        return 0.5
    return 0


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
    "according to",
    "a study",
    "a study found",
    "research shows",
    "evidence indicates",
    "suggests",
    "appears",
    "indicates",
    "in [year]",
    "historically",
    "data indicates",
    "during [event]",
    "a recent report",
    "the [organization] found",
    "a survey found",
    "statistics show",
    "the [Supreme Court/DOJ/Company] argues",
    "for example",
    "for instance",
    "such as",
    "consider",
    "namely",
    "to illustrate",
    "an example of this is",
    "over [number]%",
    "the majority of",
    "a large portion",
    "more than",
    "less than",
    "a significant number",
    "critics argue",
    "proponents claim",
    "others argue",
    "some believe that",
]

# Build trigger matcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(text) for text in trigger_phrases]
matcher.add("TriggerPhrase", patterns)


def get_external_example(essay):
    # Process text
    doc = nlp(essay)

    # Split into sentences
    sentences = list(doc.sents)

    results_dict = {}

    # Analyze each sentence
    for sent in sentences:
        sent_doc = nlp(sent.text)

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
        if (
            entity_types.intersection(
                {"PERSON", "ORG", "DATE", "EVENT", "LAW", "GPE", "WORK_OF_ART"}
            )
            or proper_noun_count >= 1
            or has_trigger
        ):
            external = True
        else:
            external = False

        results_dict[sent.text.strip()] = {
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


def parse_classify_text(items_dict):
    sorted_labels = sorted(items_dict.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_labels[0]
    second_label, second_score = sorted_labels[1]

    if top_score >= 0.65 and (top_score - second_score) >= 0.15:
        return {top_label: top_score}
    return ""


#### CHECK FOR MULTIPLE PERSPECTIVES ####
connective_categories = {
    "Contingency_Cause": [
        "because",
        "because of",
        "due to",
        "so",
        "consequently",
        "therefore",
        "thus",
        "accordingly",
        "hence",
        "as a result",
        "thereby",
        "that’s why",
    ],
    "Comparison": [
        "but",
        "however",
        "although",
        "even though",
        "though",
        "nevertheless",
        "nonetheless",
        "still",
        "yet",
        "whereas",
        "conversely",
        "on the contrary",
        "in contrast",
        "by contrast",
        "on the one hand",
        "on the other hand",
    ],
    "Expansion_Conjunction": [
        # "and",
        "also",
        "furthermore",
        "moreover",
        "additionally",
        "likewise",
        "similarly",
        "indeed",
        "besides",
        "in addition",
        "further",
    ],
    "Expansion_Instantiation": ["for example", "for instance", "such as"],
    # "Expansion_Alternative": ["or", "either"],
    "Expansion_Exception": ["except", "except that"],
    "Expansion_Specification": ["specifically", "in particular"],
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
        print(f"'{sent}' → {pred}")
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
            if similarity >= 0.8:
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
    "discover",
    "realize",
    "decide",
    "imagine",
    "believe",
    "know",
    "think",
    "guess",
    "say",
    "ask",
    "tell",
    "infer",
    "hypothesize",
    "conclude",
    "doubt",
    "interpret",
    "predict",
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
hedge_words = [
    "maybe",
    "perhaps",
    "probably",
    "likely",
    "arguably",
    "suppose",
    "seems",
    "might",
    "may",
    "suggest",
    "assume",
]

booster_words = ["definitely", "clearly", "absolutely", "certainly", "undoubtedly"]

hedging_phrases = [
    "in my opinion",
    "i think",
    "i believe",
    "it seems",
    "i don't know",
    "i guess",
    "you know",
    "as far as i know",
]


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
        for word in hedge_words:
            if word in sent_tokens:
                hedge_types.append({"type": "hedge_word", "trigger_word": word})

        # Check negated boosters
        for booster in booster_words:
            if f"not {booster}" in sent.lower() or f"n't {booster}" in sent.lower():
                hedge_types.append({"type": "booster_negated", "trigger_word": booster})

        # Check hedging phrases with Jaccard
        for phrase in hedging_phrases:
            phrase_tokens = set(phrase.split())
            sim = jaccard_similarity(sent_tokens, phrase_tokens)
            if sim >= threshold:
                hedge_types.append({"type": "hedging_phrase", "trigger_word": phrase})

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
        for booster in booster_words:
            if booster in sent.lower():
                booster_hits.append(booster)
        matches_by_sent[sent] = {
            "certainty_boosters": booster_hits,
            "booster_count": len(booster_hits),
        }
    return matches_by_sent
