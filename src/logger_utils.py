import csv
import json
from logging import Logger
import os
from typing import List


def reset_logger(logger: Logger):
    """
    Remove all the filters and the handlers from this logger

    Args:
        logger: A logger

    Returns:
        None
    """
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilter(f)


def log_csv(path: str, data: List[object]):
    """
    Append a new line to a CSV log file already exists

    Args:
        path: Path to CSV file
        data: Newline data

    Returns:
        None
    """
    with open(path, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def create_csv(path: str, header: List[str] = None):
    """
    Create a new CSV file

    Args:
        path: Path to CSV file
        header: The header of the CSV file

    Returns:
        None
    """
    if not os.path.exists(path):
        with open(path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            if header is not None:
                writer.writerow(header)


def log_json(path: str, data: dict, name: str):
    """
    Log data to a JSON file

    Args:
        path: Path to the JSON file
        data: Data to record
        name: Name of JSON file

    Returns:
        None
    """
    json_path = os.path.join(path, f'{name}.json')
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(obj=vars(data), fp=f, ensure_ascii=False, indent=4, sort_keys=True)
