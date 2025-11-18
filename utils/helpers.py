from PIL import Image
from typing import List, Optional


def extract_answer(response: str, choices: List[str]) -> Optional[str]:
    if not choices or not response:
        return None
    
    response_lower = response.lower().strip()
    response_clean = response.strip()
    
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
        if chr(65+i) == response_clean.upper() or f"{chr(65+i)}." in response_clean.upper():
            return choice
    
    return None


def format_choices(choices: List[str]) -> str:
    if not choices:
        return ""
    choices_list = [f"{i+1}. {choice}" for i, choice in enumerate(choices)]
    return "\n".join(choices_list)


def create_black_image(width: int = 512, height: int = 512) -> Image.Image:
    return Image.new('RGB', (width, height), color='black')

