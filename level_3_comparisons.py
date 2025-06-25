MIN_INSERT_WORD_COUNT = 10
MAX_INSERT_WORD_COUNT = 3
MAX_SIMILARITY_ECHO = 0.93
MAX_SIMILARITY_MINDLESS_EDIT = 0.9

IDEA_ALIGNMENT_MAX_SIMILARITY = 0.6
IDEA_ALIGNMENT_MIN_WORD_COUNT = 10

def get_mindless_echo_after_AI(action, latest_accepted_suggestion, similarity_fcn,
                               MIN_INSERT_WORD_COUNT=MIN_INSERT_WORD_COUNT, 
                               MAX_SIMILARITY_ECHO=MAX_SIMILARITY_ECHO):
    if latest_accepted_suggestion:
        first_sentence = action['action_delta'][1].split('. ')[0]
        if action['action_delta'][-1] > MIN_INSERT_WORD_COUNT:
            similarity = similarity_fcn(first_sentence, latest_accepted_suggestion)
            return similarity >= MAX_SIMILARITY_ECHO, similarity, {
                'first_sentence_written': first_sentence,
                'latest_accepted_suggestion': latest_accepted_suggestion
            }
    return False, None, ''

def get_mindless_edit_of_AI(action, latest_accepted_suggestion, similarity_fcn,
                            MAX_INSERT_WORD_COUNT=MAX_INSERT_WORD_COUNT, 
                            MAX_SIMILARITY_MINDLESS_EDIT=MAX_SIMILARITY_MINDLESS_EDIT):
    if latest_accepted_suggestion:
        if action['action_delta'][-1] <= MAX_INSERT_WORD_COUNT:
            modified_sentence = action.get('action_modified_sentences', [])
            similarity = similarity_fcn(latest_accepted_suggestion, ' '.join(modified_sentence))
            return similarity >= MAX_SIMILARITY_MINDLESS_EDIT, similarity, {
                'modified_sentence': modified_sentence,
                'latest_accepted_suggestion': latest_accepted_suggestion
            }
    return False, None, ''

def compare_sent_to_list(action_text, sent_list, similarity_fcn):
    similarities = [similarity_fcn(sent, action_text) for sent in sent_list]
    return similarities


def get_idea_alignment_order_on_AI(action, curr_idea_sentence_list, similarity_fcn):
    if isinstance(action.get('action_delta'), (list, tuple)) and len(action['action_delta']) > 1:
        inserted_text = action['action_delta'][1]
        curr_similarity = compare_sent_to_list(inserted_text, curr_idea_sentence_list, similarity_fcn)
        if any(sim > IDEA_ALIGNMENT_MAX_SIMILARITY for sim in curr_similarity):
            curr_idea_sentence_list.append(inserted_text)
            return curr_idea_sentence_list
        else:
            return [inserted_text]


def get_idea_alignment_order_on_minor_insert(action, curr_idea_sentence_list, similarity_fcn):
    if (
        isinstance(action.get('action_delta'), (list, tuple))
        and len(action['action_delta']) > 1
        and isinstance(action.get('action_modified_sentences'), list)
    ):
        if action['action_modified_sentences'] == [action['action_delta'][1]]:
            inserted_text = action['action_delta'][1]
            curr_similarity = compare_sent_to_list(inserted_text, curr_idea_sentence_list, similarity_fcn)
            if any(sim > IDEA_ALIGNMENT_MAX_SIMILARITY for sim in curr_similarity):
                return curr_idea_sentence_list, False
            else:
                return [inserted_text], True
    return curr_idea_sentence_list, False
