# Need helper functions from utils.py
import utils
import json

from sentence_transformers import SentenceTransformer, util


###### Helper Functions for Semantic Expansion #####
def get_action_expansion(
    writing_prev, writing_curr, modified_sents_count, similarity_fcn
):
    """
    Computes the semantic expansion score for a given action by measuring
    the similarity between the previous and current writing state.

    Args:
        writing_prev (str): The text before modification.
        writing_curr (str): The text after modification.
        modified_sents_count (int): The number of modified sentences.
        similarity_fcn (function): Function to compute text similarity.

    Returns:
        float: The semantic expansion score.
    """
    if modified_sents_count == 0:
        return 0
    return 1 - (similarity_fcn(writing_prev, writing_curr)) / modified_sents_count


def compute_semantic_expansion(actions_lst):
    """
    Computes the semantic expansion for all insertion-related actions
    within a session.

    Args:
        actions_lst (list): List of actions in a writing session.

    Modifies:
        Updates the 'action_semantic_expansion' field for each action in the list.
    """
    for action in actions_lst:
        action["action_semantic_expansion"] = 0

        # Only compute expansion for valid insertions
        if (
            action.get("level_1_action_type") in ["insert_text_human", "insert_text_ai"]
            and action.get("action_delta")
            and action["action_delta"][1].strip()
        ):
            modified_sents_count = len(action.get("action_modified_sentences", []))
            if modified_sents_count > 0:
                prev_writing = action.get("action_start_writing", "")
                curr_writing = action.get("action_end_writing", "")
                action["action_semantic_expansion"] = get_action_expansion(
                    prev_writing,
                    curr_writing,
                    modified_sents_count,
                    utils.get_spacy_similarity,
                )


def compute_cumulative_expansion(actions_lst):
    """
    Computes the cumulative semantic expansion score by accumulating
    semantic expansion values from all actions within a session.

    Args:
        actions_lst (list): List of actions in a writing session.

    Modifies:
        Updates the 'cumulative_semantic_expansion' field for each action in the list.
    """
    cumulative_expansion = 0
    for action in actions_lst:
        cumulative_expansion += action.get("action_semantic_expansion", 0)
        action["cumulative_semantic_expansion"] = cumulative_expansion


##### End of Semantic Expansion Helper Functions #####


##### Helper Functions for Coordination #####
def find_last_major_insert_action(actions):
    """
    Finds the last major insert action in the given list of actions.

    Args:
        actions (list): List of actions in a writing session.

    Returns:
        dict or None: The last major insert action if found, otherwise None.
    """
    for action in reversed(actions):
        if (
            "level_2_action_type" in action
            and "major_insert" in action["level_2_action_type"]
        ):
            return action
    return None


def find_last_ai_insert_suggestion(actions):
    """
    Finds the last AI-generated insert suggestion in the given list of actions.

    Args:
        actions (list): List of actions in a writing session.

    Returns:
        dict or None: The last AI insert suggestion action if found, otherwise None.
    """
    for action in reversed(actions):
        if action.get("level_1_action_type") == "insert_suggestion":
            return action
    return None


def compute_coordination(actions_lst, similarity_fcn):
    """
    Computes coordination scores by evaluating the similarity between:
    1. AI-generated insertions and the most recent human major insertion.
    2. Human major insertions and the most recent AI-generated insertion.

    Args:
        actions_lst (list): List of actions in a writing session.
        similarity_fcn (function): Function to compute text similarity.

    Modifies:
        Updates the 'coordination_score' field for each action in the list.
    """
    for idx, action in enumerate(actions_lst):
        previous_actions = actions_lst[:idx]

        level_1_action_type = action.get("level_1_action_type", "")
        level_2_action_type = action.get("level_2_action_type", "")
        action_delta = action.get("action_delta", "")

        action_start_writing = action.get("action_start_writing", "")

        # Handle AI-to-human coordination
        if level_1_action_type == "insert_suggestion" and action_delta:
            ai_inserted_text = action_delta[1]
            last_major_insert_action = find_last_major_insert_action(previous_actions)
            if last_major_insert_action and last_major_insert_action.get(
                "action_delta"
            ):
                human_inserted_text = last_major_insert_action["action_delta"][1]
                score = similarity_fcn(human_inserted_text, ai_inserted_text)
                action["coordination_score"] = [score, "AI reflects human"]

        # Handle human-to-AI coordination
        elif level_2_action_type and "major_insert" in level_2_action_type:
            last_ai_action = find_last_ai_insert_suggestion(previous_actions)
            if last_ai_action and last_ai_action.get("action_delta"):
                ai_inserted_text = last_ai_action["action_delta"][1]
                score = similarity_fcn(ai_inserted_text, action_start_writing)
                action["coordination_score"] = [score, "human reflects AI"]


##### End of Coordination Helper Functions #####


##### Helper functions for Validity #####
def get_all_writing_content(session_id):
    utils.get_filename("writing_content/", session_id)
    filename = utils.get_filename(session_id)
    writing_content = ""

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            if "content" in data:
                writing_content += data["content"]
    return writing_content


encoding_model = SentenceTransformer("all-mpnet-base-v2")


def compute_validity(session_id, actions_lst):
    """
    Computes how valid the overall human writing is compared to the relevant theoretical background.
    Adds a 'validity_score' field to each major human insertion.

    Args:
        session_id (str): Unique writing session ID
        actions_lst (list): List of action dictionaries from the writing session
        writing_content (str): Combined user-generated content

    Modifies:
        Adds 'validity_score' to actions with major human inserts.
    """
    background = (
        utils.background_info[0]
        if "antitrust" in session_id
        else utils.background_info[1]
    )
    background_embedding = encoding_model.encode(background, convert_to_tensor=True)

    for idx, action in enumerate(actions_lst):
        level_1_action_type = action.get("level_1_action_type", "")
        # action_delta = action.get("action_delta", "")
        human_sentences_temporal = action.get("human_sentences_temporal_order", "")
        # level_2_insertion_type = action.get("level_2_insertion_type", "")

        # Handle AI inserts
        if level_1_action_type == "insert_text_human" and human_sentences_temporal:
            insert = action["human_sentences_temporal_order"]
            insert_embedding = encoding_model.encode(insert, convert_to_tensor=True)
            score = util.pytorch_cos_sim(insert_embedding, background_embedding).item()
            action["validity_score"] = round(score, 4)


##### End of Validity Helper Functions #####

##### Helper functions for Confidence in Writing #####
strong_modals = [
    "must",
    "will",
    "should",
    "undoubtedly",
    "clearly",
    "definitely",
    "certainly",
    "demonstrates",
    "proves",
    "shows",
    "confirms",
    "provides evidence",
    "strongly suggests",
    "without a doubt",
    "beyond question",
    "cannot be denied",
    "is evident",
    "is obvious",
    "it is clear that",
    "is well established",
    "makes it clear",
    "compelling evidence",
    "this confirms",
    "this illustrates",
    "as a fact",
    "this reveals that",
]

stance_markers = [
    "I believe",
    "I argue",
    "I contend",
    "I maintain",
    "I suggest",
    "I assert",
    "I propose",
    "in my opinion",
    "in my view",
    "from my perspective",
    "my position is",
    "I think",
    "I have found",
    "I conclude",
    "it is my view that",
    "this analysis shows",
    "this reflects",
    "this demonstrates",
    "I interpret this as",
    "I have observed",
    "what I see is",
    "I would argue",
    "I take the position that",
    "some believe that",
    "others argue that",
    "it is widely believed that",
    "critics argue",
    "proponents claim",
    "there is a growing debate",
    "this reveals that",
    "this raises questions about",
]


def compute_confidence_linguistic(actions_lst):
    for idx, action in enumerate(actions_lst):
        level_1_action_type = action.get("level_1_action_type", "")
        level_2_insertion_type = action.get("level_2_insertion_type", "")

        if (
            level_1_action_type == "insert_text_human"
            and level_2_insertion_type
            and "major_insert" in level_2_insertion_type
        ):
            human_insert = action["human_sentences_temporal_order"]
            text_lower = human_insert.lower()
            strengths = sum(
                text_lower.count(term) for term in strong_modals + stance_markers
            )

            score = (strengths) / (len(human_insert.split()) + 1)
            action["confidence_score"] = round(score, 4)


##### End of Confidence Writing Helper Functions #####


def compute_conceptual_understanding(actions_lst):
    for idx, action in enumerate(actions_lst):
        if "validity_score" in action and "confidence_score" in action:
            action["conceptual_understanding_score"] = (
                0.6 * action["validity_score"] + 0.4 * action["confidence_score"]
            )
