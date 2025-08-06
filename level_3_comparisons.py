MIN_INSERT_WORD_COUNT = 10
MAX_INSERT_WORD_COUNT = 3
MAX_SIMILARITY_ECHO = 0.93
MAX_SIMILARITY_MINDLESS_EDIT = 0.9

IDEA_ALIGNMENT_MAX_SIMILARITY = 0.6
IDEA_ALIGNMENT_MIN_WORD_COUNT = 10


def get_mindless_echo_after_AI(
    action, past_suggestions, similarity_fcn, MAX_SIMILARITY_ECHO=MAX_SIMILARITY_ECHO
):
    copy_ai_sentences = []
    if past_suggestions:
        for sent in action["action_modified_sentences"]:
            for suggestion in past_suggestions:
                if sent != suggestion:
                    similarity = similarity_fcn(sent, suggestion)
                    if similarity >= MAX_SIMILARITY_ECHO:
                        copy_ai_sentences.append(
                            {"written_sentence": sent, "ai_suggestion": suggestion}
                        )
                        break
    if copy_ai_sentences:
        if "level_3_info" not in action:
            action["level_3_info"] = {}
        action["level_3_info"]["mindless"] = copy_ai_sentences
        return len(copy_ai_sentences)
    return 0


def compare_sent_to_list(action_text, sent_list, similarity_fcn):
    similarities = [similarity_fcn(sent, action_text) for sent in sent_list]
    return similarities


def get_idea_alignment_order_on_AI(action, curr_idea_sentence_list, similarity_fcn):
    if (
        isinstance(action.get("action_delta"), (list, tuple))
        and len(action["action_delta"]) > 1
    ):
        inserted_text = action["action_delta"][1]
        curr_similarity = compare_sent_to_list(
            inserted_text, curr_idea_sentence_list, similarity_fcn
        )
        if any(sim > IDEA_ALIGNMENT_MAX_SIMILARITY for sim in curr_similarity):
            curr_idea_sentence_list.append(inserted_text)
            return curr_idea_sentence_list
        else:
            return [inserted_text]


def get_idea_alignment_order_on_minor_insert(
    action, curr_idea_sentence_list, similarity_fcn
):
    if (
        isinstance(action.get("action_delta"), (list, tuple))
        and len(action["action_delta"]) > 1
        and isinstance(action.get("action_modified_sentences"), list)
    ):
        if action["action_modified_sentences"] == [action["action_delta"][1]]:
            inserted_text = action["action_delta"][1]
            curr_similarity = compare_sent_to_list(
                inserted_text, curr_idea_sentence_list, similarity_fcn
            )
            if any(sim > IDEA_ALIGNMENT_MAX_SIMILARITY for sim in curr_similarity):
                return curr_idea_sentence_list, False
            else:
                return [inserted_text], True
    return curr_idea_sentence_list, False
