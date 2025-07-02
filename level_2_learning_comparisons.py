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


from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def classify_text(text):
    result = classifier(
        text,
        candidate_labels=["question", "imperative", "statement"],
    )
    return {
        "question": result["scores"][0],
        "imperative": result["scores"][1],
        "statement": result["scores"][2],
    }
