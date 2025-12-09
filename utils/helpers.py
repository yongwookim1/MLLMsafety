"""
Helper functions for text processing and answer extraction.

This module provides utilities for working with multiple-choice questions
and answer validation in evaluation tasks.
"""

from PIL import Image
from typing import List, Optional, Tuple
import random


def extract_answer(response: str, choices: List[str]) -> Optional[str]:
    """
    Extract the selected answer from a model response for multiple-choice questions.

    This function handles various formats of answer indication:
    - Letter labels (A, B, C, etc.)
    - Numbered choices (1, 2, 3, etc.)
    - Exact text matches
    - Partial text matches

    Args:
        response: Raw text response from the model
        choices: List of possible answer choices

    Returns:
        The matched choice text, or None if no match found
    """
    if not choices or not response:
        return None

    # Clean and normalize the response
    response_clean = response.strip()
    response_upper = response_clean.upper()

    # Check for letter-based answers (A, B, C, etc.)
    for i, choice in enumerate(choices):
        letter_label = chr(65 + i)  # A=65, B=66, etc.

        # Check various letter formats: "A", "A.", "A)"
        if (response_upper and letter_label == response_upper[0]) or \
            f"{letter_label}." in response_upper or \
            f"{letter_label})" in response_upper:
            return choice

    # Prepare lowercase versions for text matching
    response_lower = response.lower().strip()

    # Check for exact or partial text matches
    for choice in choices:
        choice_lower = choice.lower().strip()
        choice_clean = choice.strip()

        # Exact matches (case-insensitive)
        if choice_clean == response_clean or choice_lower == response_lower:
            return choice

        # Partial matches (response contains choice or vice versa)
        if choice_clean in response_clean or response_clean in choice_clean:
            return choice

        # Overlapping text matches
        if choice_lower in response_lower or response_lower in choice_lower:
            return choice

    # Check for numbered answers (1, 2, 3, etc.)
    for i, choice in enumerate(choices):
        number = i + 1

        if f"{number}." in response or f"{number})" in response:
            return choice

        if str(number) == response_clean:
            return choice

    return None


def format_choices(choices: List[str], shuffle: bool = True) -> Tuple[str, List[str]]:
    """
    Format multiple-choice options with letter labels.

    Args:
        choices: List of choice texts
        shuffle: Whether to randomize the order of choices

    Returns:
        Tuple of (formatted_string, shuffled_choices_list)
    """
    if not choices:
        return "", []

    # Create a copy to avoid modifying the original
    formatted_choices = choices.copy()

    # Shuffle if requested (and there are multiple choices)
    if shuffle and len(formatted_choices) > 1:
        random.shuffle(formatted_choices)

    # Format with letter labels (A, B, C, etc.)
    letter_labels = ["A", "B", "C"]
    labeled_choices = [
        f"{letter_labels[i]}. {choice}"
        for i, choice in enumerate(formatted_choices)
    ]

    return "\n".join(labeled_choices), formatted_choices


def create_black_image(width: int = 512, height: int = 512) -> Image.Image:
    """
    Create a solid black image of specified dimensions.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        PIL Image object filled with black color
    """
    return Image.new('RGB', (width, height), color='black')

