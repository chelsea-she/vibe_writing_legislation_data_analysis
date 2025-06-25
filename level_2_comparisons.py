# Need helper functions from utils.py
import utils

MIN_INSERT_WORD_COUNT = 10
MAJOR_INSRT_MAX_SIMILARITY = 0.9
MINOR_INSRT_MAX_SIMILARITY = 0.95
MAX_SIMILARITY = 0.95


def get_similarity_with_prev_writing_for_level_2(action, prev_writing, similarity_fcn):
    prev_sents = utils.sent_tokenize(prev_writing)
    curr_sents = utils.sent_tokenize(action["action_end_writing"])
    select_sents_after_action = action["action_modified_sentences"]
    select_sents_before_action = []
    for sent in prev_sents:
        if sent not in curr_sents:
            select_sents_before_action.append(sent)
    similarity = abs(
        similarity_fcn(
            " ".join(select_sents_after_action), " ".join(select_sents_before_action)
        )
    )
    return similarity, {
        "select_sents_before_action": select_sents_before_action,
        "select_sents_after_action": select_sents_after_action,
    }


def parse_level_2_major_insert_major_semantic_diff(
    action,
    prev_writing_similarity,
    MIN_INSERT_WORD_COUNT=MIN_INSERT_WORD_COUNT,
    MAJOR_INSRT_MAX_SIMILARITY=MAJOR_INSRT_MAX_SIMILARITY,
):
    if (
        action["action_type"] == "insert_text"
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
        action["action_type"] == "insert_text"
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
        action["action_type"] == "insert_text"
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
        action["action_type"] == "insert_text"
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
