# Need helper functions from utils.py
import utils

def apply_text_operations(doc, mask, ops, source, debug=False):
    """
    Applies a sequence of text operations (retain, insert, delete) to a document.

    Args:
        doc (str): The original text document.
        mask (str): The mask representing the source of modifications.
        ops (list): A list of operations containing 'retain', 'insert', or 'delete' actions.
        source (str): The source of the modification (e.g., 'api' or 'user').
        debug (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        tuple: 
            - str: The updated document after applying the operations.
            - str: The updated mask reflecting the modifications.
    """
    original_doc = doc
    original_mask = mask
    new_doc = ''
    new_mask = ''

    for _, op in enumerate(ops):
        if 'retain' in op:
            num_char = op['retain']
            if debug:
                print('@ Retain:', num_char)
            retain_doc = original_doc[:num_char]
            retain_mask = original_mask[:num_char]
            original_doc = original_doc[num_char:]
            original_mask = original_mask[num_char:]
            new_doc = new_doc + retain_doc
            new_mask = new_mask + retain_mask

        elif 'insert' in op:
            insert_doc = op['insert']
            if debug:
                print('@ Insert:', insert_doc)
            if source == 'api':
                # Use '*' for API insertions
                insert_mask = '*' * len(insert_doc)
            else:
                # Use '_' for user insertions
                insert_mask = '_' * len(insert_doc)

            if isinstance(insert_doc, dict):
                if 'image' in insert_doc:
                    print('Skipping invalid object insertion (image)')
                else:
                    import ipdb
                    ipdb.set_trace()
                    print(insert_doc)
            else:
                new_doc = new_doc + insert_doc
                new_mask = new_mask + insert_mask

        elif 'delete' in op:
            num_char = op['delete']
            if debug:
                print('@ Delete:', num_char)
            if original_doc:
                original_doc = original_doc[num_char:]
                original_mask = original_mask[num_char:]
            else:
                new_doc = new_doc[:-num_char]
                new_mask = new_mask[:-num_char]

        else:
            print('@ Unknown operation:', op)

        if debug:
            print('Document:', new_doc + original_doc, '\n')

    if debug:
        print('Final document:', new_doc + original_doc)

    return new_doc + original_doc, new_mask + original_mask

def apply_logs_to_writing(current_writing, current_mask, all_logs):
    """
    Applies a series of logged text operations to a given document.

    Args:
        current_writing (str): The current state of the document before applying logs.
        current_mask (str): The corresponding mask tracking source modifications.
        all_logs (list): A list of logs containing 'textDelta' operations.

    Returns:
        tuple: 
            - str: The modified document after applying all logged operations.
            - str: The updated mask reflecting changes.
    """
    for log in all_logs:
        if 'textDelta' in log and 'ops' in log['textDelta']:
            ops = log['textDelta']['ops']
            source = log['eventSource']
            current_writing, current_mask = apply_text_operations(
                current_writing, current_mask, ops, source
            )
    return current_writing, current_mask

def convert_last_action_to_complete_action(last_action):
    """
    Converts the last action into a complete action dictionary for live analysis.
    
    Args:
        last_action (dict): The last recorded action.
    
    Returns:
        dict: A complete action dictionary with additional keys populated.
    """
    action = last_action.copy()  # Create a shallow copy to avoid modifying the original
    action['action_end_time'] = utils.convert_timestamp_to_string(
        utils.get_timestamp(last_action['action_logs'][-1]['eventTimestamp'])
    )
    action['action_delta'] = last_action['delta_at_save']
    action['action_end_writing'] = last_action['writing_at_save']
    action['action_end_mask'] = last_action['mask_at_save']
    return action

def get_action_type_from_log(log):
    """
    Determines the type of action and whether the writing was modified based on a log entry.

    Args:
        log (dict): A single log entry containing details such as 'eventSource', 'eventName', 
                    and optionally 'textDelta' for text-related changes.

    Returns:
        tuple: (action_type, writing_was_modified)
            - action_type (str): The type of action derived from the log. Possible values include:
                'present_suggestion', 'query_suggestion', 'insert_text', 'delete_text', 
                'accept_suggestion', 'reject_suggestion', 'cursor_operation', etc.
            - writing_was_modified (bool): True if the log indicates that the writing was modified 
                                            (e.g., through 'textDelta' operations), False otherwise.

    Raises:
        None, but prints error messages for unrecognized 'eventName' or 'eventSource'.
    """
    action_type = None
    writing_was_modified = False
    if log['eventSource'] == 'api':
        if log['eventName'] == 'suggestion-open' or \
            log['eventName'] == 'suggestion-reopen' or \
            log['eventName'] == 'suggestion-close' or \
            log['eventName'] in ['cursor-forward', 'cursor-backward', 'cursor-select']:
            action_type = 'present_suggestion'
        elif log['eventName'] == 'text-insert':
            action_type = 'insert_suggestion'
        else:
            print('Error: {}'.format(log))
    elif log['eventSource'] == 'user':
        if log['eventName'] == 'suggestion-get':
            action_type = 'query_suggestion'
        elif log['eventName'] in ['suggestion-hover', 'suggestion-up', 'suggestion-down']:
            action_type = 'hover_over_text'
        elif log['eventName'] == 'suggestion-select':
            action_type = 'accept_suggestion'
        elif log['eventName'] == 'suggestion-close':
            action_type = 'reject_suggestion'
        elif log['eventName'] == 'suggestion-reopen':
            action_type = 'present_suggestion'
        elif log['eventName'] in ['cursor-select', 'cursor-forward', 'cursor-backward']:
            action_type = 'cursor_operation'
        elif log['eventName'] == 'text-insert':
            action_type = 'insert_text'
        elif log['eventName'] == 'text-delete':
            action_type = 'TBD'
        else:
            print('Error: {}'.format(log))
    else:
        print('Error: {}'.format(log))
    if 'textDelta' in log and 'ops' in log['textDelta']:
        writing_was_modified = True
    if action_type is None:
        print(log['eventName'])
    return action_type, writing_was_modified

def extract_and_clean_text_modifications_from_action(writing_at_start_of_action, 
                                                     current_logs, current_action):
    """
    Extracts text modifications (insertions/deletions) from logs into cohesive operations.
    
    Args:
        writing_at_start_of_action (str): The text at the start of the action.
        current_logs (list): Logs associated with the action.
        current_action (str): The type of the current action.

    Returns:
        tuple: (operation_type, modified_text, char_count, word_count)
            - operation_type: 'INSERT' or 'DELETE'.
            - modified_text: The text that was inserted or deleted.
            - char_count: The number of characters modified.
            - word_count: The number of words modified.
    """
    original_text = writing_at_start_of_action
    new_text = ''
    insert_string = ''
    insert_char_count = 0
    delete_string = ''
    delete_char_count = 0

    for log in current_logs:
        if 'textDelta' in log and 'ops' in log['textDelta']:
            ops = log['textDelta']['ops']
            source = log['eventSource']
            for op in ops:
                if 'retain' in op:
                    num_char = op['retain']
                    retain_text = original_text[:num_char]
                    original_text = original_text[num_char:]
                    new_text = new_text + retain_text
                if 'insert' in op:
                    insert_text = op['insert']
                    insert_char_count += len(insert_text)
                    new_text = new_text + insert_text
                    insert_string = insert_string + insert_text
                if 'delete' in op:
                    num_char = op['delete']
                    delete_char_count += num_char
                    if original_text:
                        delete_string = original_text[:num_char] + delete_string
                        original_text = original_text[num_char:]
                    else:
                        delete_string = new_text[-num_char:] + delete_string
                        new_text = new_text[:-num_char]
                        insert_string = insert_string[:-num_char]

    if current_action in ['insert_text', 'insert_suggestion']:
        return ('INSERT', insert_string, len(insert_string), len(insert_string.split()))
    elif current_action in ['delete_text', 'TBD']:
        return ('DELETE', delete_string, len(delete_string), len(delete_string.split()))
    else:
        return ''
    