# To install the pandas library, run the following command in your terminal:
# pip install pandas

# To execute this script, run the following command in your terminal:
# python extract_coauthor_raw_logs.py

import pandas as pd
import json
import os

# The directory of the CSV files (argumentative metadata and creative metadata) is assumed to
# be the same as this Python script.
# Please update the `script_dir` variable if the files are located elsewhere
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to the metadata CSV files
csv_path = os.path.join(script_dir, "legislation_pilot1.csv")

# Load the metadata from the CSV files
legislation_csv = pd.read_csv(csv_path)

# Define the prompts for extracting session IDs
# Please choose whatever prompts you want
legislation_prompts = ["corporate", "antitrust"]

jsonl_names = {}


def extract_session_ids(df, prompts, prefix):
    """
    Extracts session IDs from a DataFrame based on prompt codes and organizes them by a prefix.

    Args:
        df (pandas.DataFrame): DataFrame containing session data with columns 'prompt_code' and 'session_id'.
        prompts (list): List of prompt codes to filter session IDs.
        prefix (str): Prefix to use for naming keys in the output dictionary.

    Returns:
        None: Updates the global dictionary 'jsonl_names' by adding session IDs grouped by prompts.

    Example:
        If 'prefix' is "creative" and a prompt is "mana", the resulting key will be "creative_mana".

    Notes:
        - Assumes 'jsonl_names' is defined globally.
        - Filters session IDs where 'prompt_code' matches any value in the 'prompts' list.
    """
    for prompt in prompts:
        matching_ids = df[df["prompt_code"] == prompt]["session_id"].tolist()
        key = f"{prefix}_{prompt}"
        jsonl_names[key] = matching_ids


# Extract session IDs for prompts
extract_session_ids(legislation_csv, legislation_prompts, "legislation")


def get_logs(jsonl_names_dict, logs_folder_path):
    """
    Reads and processes JSONL files containing session logs.

    Args:
        jsonl_names_dict (dict): A dictionary where keys are session prefixes (e.g., "creative_mana")
                                 and values are lists of session IDs.
        logs_folder_path (str): Path to the folder containing the JSONL log files.

    Returns:
        dict: A dictionary where keys are unique session identifiers (e.g., "creative_mana_1") and values
              are lists of filtered logs for each session.

    Raises:
        FileNotFoundError: If a specified JSONL file is not found in the given folder path.
        json.JSONDecodeError: If a file cannot be decoded as valid JSONL.
        Exception: For any other unexpected errors during file processing.

    Notes:
        - Filters out events with the eventName "system-initialize".
        - Prints error messages for files that cannot be found or processed.
    """
    result = {}
    for key, session_ids in jsonl_names_dict.items():
        for i, session_id in enumerate(session_ids):
            # Construct the filename for the current session
            filename = f"{session_id}.jsonl"
            filepath = os.path.join(logs_folder_path, filename)
            try:
                # Read and parse the JSONL file
                with open(filepath, "r") as file:
                    raw_logs = [json.loads(line) for line in file]
                # Filter out 'system-initialize' events
                filtered_logs = [
                    log
                    for log in raw_logs
                    if log.get("eventName") != "system-initialize"
                ]
                # Create a unique key for each session and store the parsed logs
                result[f"{key}_{i + 1}"] = filtered_logs
            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filepath}: {e}")
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")
    return result


# Define the path to the folder containing JSONL files
# The 'coauthor-v1.0' folder was downloaded directly from the CoAuthor website.
folder_path_to_logs = os.path.join(script_dir, "pilot1_raw_logs")

# Define the output path for raw logs
output_file_path = os.path.join(script_dir, "pilot1_logs.json")

try:
    # Get raw logs
    logs = get_logs(jsonl_names, folder_path_to_logs)

    # Save the extracted logs to a JSON file in the same folder as the CSV files
    with open(output_file_path, "w") as json_file:
        json.dump(logs, json_file, indent=4)
    # Print this statement when all logs were successfully saved
    print(f"Logs successfully saved to: {output_file_path}")

except FileNotFoundError as e:
    print(e)

except Exception as e:
    print(f"An unexpected error occurred: {e}")
