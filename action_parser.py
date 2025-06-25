# Need helper functions from utils.py and parser_helper.py
# utils.py contains basic utilities such as timestamp conversion, sentence tokenization, and etc
# parser_helper.py contains helper functions for parsing raw logs needed in the parser class
import utils
import parser_helper

# Level 1: Parse the raw logs into a structured format for further analysis.
# This process includes merging individual action logs (e.g., inserting or deleting 
# single letters) into cohesive words and sentences while tracking modified sentences.
class MergeActionsAnalyzer:
    """
    Parses raw logs into structured actions. Finalizes the current action when encountering 
    large deletions, source changes, or significant action type transitions. Otherwise, 
    merges consecutive edits into ongoing insertions.
    """

    def __init__(self, last_action, raw_logs=None, actions_list=None):
        """
        Initializes the analyzer with logs or an actions list.

        Args:
            last_action (dict): The last recorded action to continue parsing.
            raw_logs (list, optional): Raw logs to parse. Defaults to None.
            actions_list (list, optional): Existing actions list to update. Defaults to None.
        """
        if raw_logs is None:
            assert actions_list is not None and last_action is not None
            latest_action = parser_helper.convert_last_action_to_complete_action(last_action)
            actions_list.append(latest_action)
            self.actions_lst = actions_list
            self.last_action = last_action
            self.analyzer_on = True
        else:
            self.actions_lst, self.last_action = self.parse_actions_from_logs(
                raw_logs, last_action
            )
            self.analyzer_on = False

    def parse_actions_from_logs(self, all_logs, last_action=None, DLT_CHAR_MAX_COUNT=9):
        """
        Parse actions from raw logs with regular merge logic.

        Args:
            all_logs (list): The raw logs to parse.
            last_action (dict, optional): The last recorded action. Defaults to None.
            DLT_CHAR_MAX_COUNT (int): The maximum number of characters for a "tiny delete."

        Returns:
            tuple: (all_actions_lst, last_action)
        """
        all_actions_lst = []

        # Initialize if no last action
        if last_action is None:
            log_action, writing_modified = parser_helper.get_action_type_from_log(all_logs[0])
            if log_action == 'TBD':
                log_action = 'insert_text'
            last_action = {
                'action_type': log_action,
                'action_source': all_logs[0]['eventSource'],
                'action_logs': [all_logs[0]],
                'action_start_log_id': 0,
                'action_start_time': utils.convert_timestamp_to_string(
                    utils.get_timestamp(all_logs[0]['eventTimestamp'])
                ),
                'action_start_writing': '',
                'action_start_mask': '',
                'writing_modified': writing_modified,
                'sentences_seen_so_far': {}
            }

        # Unpack last action
        current_action = last_action['action_type']
        current_source = last_action['action_source']
        current_logs = last_action['action_logs']
        current_start_time = utils.convert_string_to_timestamp(last_action['action_start_time'])
        current_writing = last_action['action_start_writing']
        current_mask = last_action['action_start_mask']
        prev_writing_modified = last_action['writing_modified']
        start_log_id = last_action['action_start_log_id']
        sentences_seen_so_far = last_action['sentences_seen_so_far']

        action_start_writing_for_current_action = current_writing

        action_modified_sentences = []
        sentences_temporal_order = []
        start_id = start_log_id + len(current_logs)

        # Main section of this parser => loop through raw logs
        for i in range(start_id, len(all_logs)):
            log = all_logs[i]
            log_action, writing_modified = parser_helper.get_action_type_from_log(log)

            # Decide if we need to finalize the current action
            if log['eventSource'] != current_source:
                # Source changes => finalize
                to_end_action = True
            elif log_action == current_action:
                # Same action => accumulate
                to_end_action = False
            elif log_action != current_action and current_source == 'api':
                # API special case
                to_end_action = False
                if current_action == 'present_suggestion' and log_action == 'insert_suggestion':
                    current_action = log_action
            elif (log_action == 'cursor_operation'
                  and current_action == 'insert_text'
                  and current_source == 'user'):
                # Cursor operation while inserting => keep merging
                to_end_action = False
                log_action = current_action
            elif (log_action == 'delete_text'
                  and current_action == 'insert_text'):
                # Delete while inserting => keep merging if not huge
                deleted_now = 0
                if 'textDelta' in log and 'ops' in log['textDelta']:
                    for op_dct in log['textDelta']['ops']:
                        if 'delete' in op_dct:
                            deleted_now += op_dct['delete']
                if deleted_now <= DLT_CHAR_MAX_COUNT:
                    to_end_action = False
                    log_action = current_action
                else:
                    to_end_action = True
            elif (log_action != 'TBD'
                  and log_action != current_action
                  and current_source == 'user'):
                # Different user action => finalize
                to_end_action = True
            elif log_action == 'TBD':
                # If 'TBD', handle merges or finalize
                if current_action not in ['insert_text', 'delete_text']:
                    to_end_action = True
                    log_action = 'delete_text'
                elif current_action == 'delete_text':
                    # Already deleting => keep merging
                    to_end_action = False
                    log_action = 'delete_text'
                else:
                    # Check if small deletion => could still be part of an insertion
                    latest_delete_logs_start_id = len(current_logs)
                    delete_char_count = 0
                    for j in range(len(current_logs), 0, -1):
                        if latest_delete_logs_start_id - j == 1:
                            lg = current_logs[j]
                            if (lg['eventName'] == 'text-delete' and 'textDelta' in lg):
                                for op_dct in lg['textDelta']['ops']:
                                    if 'delete' in op_dct:
                                        delete_char_count += op_dct['delete']
                                        latest_delete_logs_start_id = j
                    assert delete_char_count <= DLT_CHAR_MAX_COUNT
                    if 'textDelta' in log:
                        for op_dct in log['textDelta']['ops']:
                            if 'delete' in op_dct:
                                delete_char_count += op_dct['delete']

                    # If it's large, finalize current insertion
                    if delete_char_count > DLT_CHAR_MAX_COUNT:
                        pre_action_writing_for_partial = current_writing

                        delta = parser_helper.extract_and_clean_text_modifications_from_action(
                            pre_action_writing_for_partial,
                            current_logs[:latest_delete_logs_start_id],
                            current_action
                        )
                        post_action_writing_for_partial, post_action_mask_for_partial = (
                            parser_helper.apply_logs_to_writing(
                                pre_action_writing_for_partial,
                                current_mask,
                                current_logs[:latest_delete_logs_start_id]
                            )
                        )
                        current_sentences = {}
                        for sent in utils.sent_tokenize(post_action_writing_for_partial):
                            if sent not in sentences_seen_so_far:
                                sentences_seen_so_far[sent] = len(sentences_seen_so_far)
                                action_modified_sentences.append(sent)
                            current_sentences[sent] = sentences_seen_so_far[sent]
                        sentences_temporal_order = [
                            x[0] for x in sorted(current_sentences.items(), key=lambda t: t[1])
                        ]

                        # Finalize the partial insertion
                        action_dct = {
                            'action_type': current_action,
                            'action_source': current_source,
                            'action_logs': current_logs[:latest_delete_logs_start_id],
                            'action_start_log_id': start_log_id,
                            'action_start_time': utils.convert_timestamp_to_string(current_start_time),
                            'action_start_writing': pre_action_writing_for_partial,
                            'action_end_time': utils.convert_timestamp_to_string(
                                utils.get_timestamp(current_logs[latest_delete_logs_start_id - 1]['eventTimestamp'])
                            ),
                            'action_end_writing': post_action_writing_for_partial,
                            'action_end_mask': post_action_mask_for_partial,
                            'writing_modified': prev_writing_modified,
                            'action_delta': delta,
                            'action_modified_sentences': action_modified_sentences,
                            'sentences_temporal_order': sentences_temporal_order
                        }
                        all_actions_lst.append(action_dct)

                        # Start a new delete action
                        to_end_action = False
                        current_action = 'delete_text'
                        
                        current_writing = post_action_writing_for_partial
                        current_mask = post_action_mask_for_partial

                        if latest_delete_logs_start_id == len(current_logs):
                            current_logs = [log]
                        else:
                            current_logs = current_logs[latest_delete_logs_start_id:] + [log]
                        current_start_time = utils.get_timestamp(current_logs[0]['eventTimestamp'])
                        prev_writing_modified = True
                        start_log_id += latest_delete_logs_start_id
                        action_modified_sentences = []
                        action_start_writing_for_current_action = current_writing
                    else:
                        to_end_action = False
                        log_action = current_action
            else:
                print('Error: {}, {}, {}'.format(current_action, log_action, log))
                to_end_action = False

            # Finalize if needed
            if to_end_action:
                if current_action == 'TBD':
                    total_deleted = 0
                    for clg in current_logs:
                        if clg.get('eventName') == 'text-delete' and 'textDelta' in clg:
                            for op_dct in clg['textDelta']['ops']:
                                if 'delete' in op_dct:
                                    total_deleted += op_dct['delete']
                    if total_deleted <= DLT_CHAR_MAX_COUNT:
                        current_action = 'insert_text'
                    else:
                        current_action = 'delete_text'

                pre_action_writing_for_finalize = action_start_writing_for_current_action

                if prev_writing_modified:
                    delta = parser_helper.extract_and_clean_text_modifications_from_action(
                        pre_action_writing_for_finalize, current_logs, current_action
                    )
                    post_action_writing_for_finalize, post_action_mask_for_finalize = (
                        parser_helper.apply_logs_to_writing(
                            pre_action_writing_for_finalize,
                            current_mask,
                            current_logs
                        )
                    )
                    current_sentences = {}
                    for sent in utils.sent_tokenize(post_action_writing_for_finalize):
                        if sent not in sentences_seen_so_far:
                            sentences_seen_so_far[sent] = len(sentences_seen_so_far)
                            action_modified_sentences.append(sent)
                        current_sentences[sent] = sentences_seen_so_far[sent]
                    sentences_temporal_order = [
                        x[0] for x in sorted(current_sentences.items(), key=lambda t: t[1])
                    ]
                else:
                    delta = ''

                    post_action_writing_for_finalize = pre_action_writing_for_finalize
                    post_action_mask_for_finalize = current_mask

                # Finalize the action
                action_dct = {
                    'action_type': current_action,
                    'action_source': current_source,
                    'action_logs': current_logs,
                    'action_start_log_id': start_log_id,
                    'action_start_time': utils.convert_timestamp_to_string(current_start_time),
                    'action_start_writing': pre_action_writing_for_finalize, 
                    'action_end_time': utils.convert_timestamp_to_string(
                        utils.get_timestamp(current_logs[-1]['eventTimestamp'])
                    ),
                    'action_end_writing': post_action_writing_for_finalize, 
                    'action_end_mask': post_action_mask_for_finalize,
                    'writing_modified': prev_writing_modified,
                    'action_delta': delta,
                    'action_modified_sentences': action_modified_sentences,
                    'sentences_temporal_order': sentences_temporal_order
                }
                all_actions_lst.append(action_dct)

                # Begin a new action with the current log
                current_action = log_action
                current_source = log['eventSource']
                current_logs = [log]
                current_start_time = utils.get_timestamp(log['eventTimestamp'])
                writing_modified = last_action['writing_modified']
                start_log_id = i
                action_modified_sentences = []

                current_writing = post_action_writing_for_finalize
                current_mask = post_action_mask_for_finalize

                action_start_writing_for_current_action = current_writing
                to_end_action = False
            else:
                # Continue accumulating logs
                current_logs.append(log)

            if not prev_writing_modified:
                prev_writing_modified = writing_modified

        # After looping, finalize leftover
        if current_action == 'TBD':
            total_deleted = 0
            for clg in current_logs:
                if clg.get('eventName') == 'text-delete' and 'textDelta' in clg:
                    for op_dct in clg['textDelta']['ops']:
                        if 'delete' in op_dct:
                            total_deleted += op_dct['delete']
            if total_deleted <= DLT_CHAR_MAX_COUNT:
                current_action = 'insert_text'
            else:
                current_action = 'delete_text'

        pre_action_writing_for_finalize = action_start_writing_for_current_action
        if prev_writing_modified:
            last_delta = parser_helper.extract_and_clean_text_modifications_from_action(
                pre_action_writing_for_finalize, current_logs, current_action
            )
            last_writing, last_mask = parser_helper.apply_logs_to_writing(
                pre_action_writing_for_finalize, current_mask, current_logs
            )
            last_sentences = {}
            sid = len(sentences_seen_so_far)
            for sent in utils.sent_tokenize(last_writing):
                if sent not in sentences_seen_so_far:
                    last_sentences[sent] = sid
                    sid += 1
                    action_modified_sentences.append(sent)
                else:
                    last_sentences[sent] = sentences_seen_so_far[sent]
            sentences_temporal_order = [
                x[0] for x in sorted(last_sentences.items(), key=lambda t: t[1])
            ]
        else:
            last_writing = pre_action_writing_for_finalize
            last_mask = current_mask
            last_delta = ''

        action_dct = {
            'action_type': current_action,
            'action_source': current_source,
            'action_logs': current_logs,
            'action_start_log_id': start_log_id,
            'action_start_time': utils.convert_timestamp_to_string(current_start_time),
            'action_start_writing': pre_action_writing_for_finalize, 
            'action_end_time': utils.convert_timestamp_to_string(
                utils.get_timestamp(current_logs[-1]['eventTimestamp'])
            ) if current_logs else utils.convert_timestamp_to_string(current_start_time),
            'action_end_writing': last_writing,  
            'action_end_mask': last_mask,
            'writing_modified': prev_writing_modified,
            'action_delta': last_delta,
            'action_modified_sentences': action_modified_sentences,
            'sentences_temporal_order': sentences_temporal_order
        }
        all_actions_lst.append(action_dct)

        last_action = {
            'action_type': current_action,
            'action_source': current_source,
            'action_logs': current_logs,
            'action_start_log_id': start_log_id,
            'action_start_time': utils.convert_timestamp_to_string(current_start_time),
            'action_start_writing': last_writing,
            'action_start_mask': last_mask,
            'writing_modified': prev_writing_modified,
            'writing_at_save': last_writing,
            'mask_at_save': last_mask,
            'delta_at_save': last_delta,
            'sentences_seen_so_far': sentences_seen_so_far,
            'action_modified_sentences': action_modified_sentences,
            'sentences_temporal_order': sentences_temporal_order
        }

        return all_actions_lst, last_action
