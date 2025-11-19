from PIL import Image
from typing import List, Optional, Tuple
import random


def extract_answer(response: str, choices: List[str]) -> Optional[str]:
    if not choices or not response:
        return None
    
    response_clean = response.strip()
    response_upper = response_clean.upper()
    
    for i, choice in enumerate(choices):
        label = chr(65 + i)
        if label == response_upper[0] or f"{label}." in response_upper or f"{label})" in response_upper:
            return choice
    
    response_lower = response.lower().strip()
    
    for choice in choices:
        choice_lower = choice.lower().strip()
        choice_clean = choice.strip()
        
        if choice_clean == response_clean or choice_lower == response_lower:
            return choice
        
        if choice_clean in response_clean or response_clean in choice_clean:
            return choice
        
        if choice_lower in response_lower or response_lower in choice_lower:
            return choice
    
    for i, choice in enumerate(choices):
        if f"{i+1}." in response or f"{i+1})" in response:
            return choice
        if str(i+1) == response_clean:
            return choice
    
    return None


def format_choices(choices: List[str], shuffle: bool = True) -> Tuple[str, List[str]]:
    if not choices:
        return "", []
    
    shuffled_choices = choices.copy()
    if shuffle and len(shuffled_choices) > 1:
        random.shuffle(shuffled_choices)
    
    labels = ["A", "B", "C"]
    choices_list = [f"{labels[i]}. {choice}" for i, choice in enumerate(shuffled_choices)]
    return "\n".join(choices_list), shuffled_choices


def create_black_image(width: int = 512, height: int = 512) -> Image.Image:
    return Image.new('RGB', (width, height), color='black')

