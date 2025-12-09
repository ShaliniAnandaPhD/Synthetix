"""
Lexical Injector for adding city-specific flavor phrases to text.
"""

import json
import os
import random
import re
from typing import Dict, Any, List, Optional


def _load_city_profiles(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load city profiles from JSON file.
    
    Args:
        config_path: Path to city_profiles.json. Defaults to config/city_profiles.json.
    
    Returns:
        Dictionary of city profiles.
    """
    if config_path is None:
        # Get the project root by going up from src/core/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        config_path = os.path.join(project_root, 'config', 'city_profiles.json')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"City profiles config not found at: {config_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in city profiles: {e.msg}", e.doc, e.pos)


def inject_flavor(text: str, city_name: str, config_path: Optional[str] = None) -> str:
    """
    Inject city-specific lexical flavor into text based on injection rate.
    
    This function reads the lexical_style configuration for the given city
    and randomly inserts phrases from the phrases list based on the injection_rate.
    
    Args:
        text: The input text to inject phrases into
        city_name: Name of the city (e.g., 'Kansas City', 'Philadelphia')
        config_path: Optional path to city_profiles.json
    
    Returns:
        Text with injected flavor phrases.
    
    Raises:
        KeyError: If city not found in profiles.
    """
    # Load profiles
    profiles = _load_city_profiles(config_path)
    
    if city_name not in profiles:
        available_cities = ', '.join(profiles.keys())
        raise KeyError(
            f"City '{city_name}' not found. Available cities: {available_cities}"
        )
    
    profile = profiles[city_name]
    
    # Get lexical style configuration with graceful defaults
    lexical_style = profile.get('lexical_style', {})
    injection_rate = lexical_style.get('injection_rate', 0.0)
    phrases = lexical_style.get('phrases', [])
    
    # If no injection or no phrases, return original text
    if injection_rate <= 0 or not phrases:
        return text
    
    # Split text into sentences (simple split on . ! ?)
    sentence_endings = r'([.!?])\s+'
    sentences = re.split(sentence_endings, text)
    
    # Reconstruct sentences with their punctuation
    reconstructed = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        # Check if next item is punctuation
        if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1
        
        if sentence.strip():
            reconstructed.append(sentence)
    
    # If we couldn't split into sentences, treat as one sentence
    if not reconstructed:
        reconstructed = [text]
    
    # Inject phrases based on injection_rate
    result_sentences = []
    
    for sentence in reconstructed:
        # Roll for injection
        if random.random() < injection_rate:
            injected_sentence = _inject_phrase_into_sentence(sentence, phrases)
            result_sentences.append(injected_sentence)
        else:
            result_sentences.append(sentence)
    
    # Join sentences back together
    return ' '.join(result_sentences)


def _inject_phrase_into_sentence(sentence: str, phrases: List[str]) -> str:
    """
    Inject a random phrase into a sentence naturally.
    
    Args:
        sentence: The sentence to inject into
        phrases: List of possible phrases to inject
    
    Returns:
        Sentence with injected phrase.
    """
    if not phrases:
        return sentence
    
    # Choose a random phrase
    phrase = random.choice(phrases)
    
    # Determine injection strategy randomly
    strategies = ['prefix', 'suffix', 'parenthetical', 'emphasis']
    strategy = random.choice(strategies)
    
    sentence = sentence.strip()
    
    if strategy == 'prefix':
        # Add as prefix: "You know, [phrase] - [sentence]"
        return f"You know, {phrase} - {sentence}"
    
    elif strategy == 'suffix':
        # Add as suffix: "[sentence] - that's [phrase] right there."
        # Remove ending punctuation temporarily
        ending_punct = ''
        if sentence and sentence[-1] in '.!?':
            ending_punct = sentence[-1]
            sentence = sentence[:-1]
        return f"{sentence} - that's {phrase} right there{ending_punct}"
    
    elif strategy == 'parenthetical':
        # Add parenthetically in the middle
        words = sentence.split()
        if len(words) > 3:
            # Insert after first few words
            insert_pos = min(3, len(words) // 2)
            words.insert(insert_pos, f"({phrase})")
            return ' '.join(words)
        else:
            return f"{sentence} ({phrase})"
    
    else:  # emphasis
        # Capitalize and emphasize: "[sentence] [PHRASE]!"
        ending_punct = ''
        if sentence and sentence[-1] in '.!?':
            ending_punct = sentence[-1]
            sentence = sentence[:-1]
        
        # If ending is not emphatic, make it so
        if ending_punct != '!':
            ending_punct = '!'
        
        return f"{sentence} {phrase.upper()}{ending_punct}"


def get_formality_level(city_name: str, config_path: Optional[str] = None) -> float:
    """
    Get the formality level for a city.
    
    Args:
        city_name: Name of the city
        config_path: Optional path to city_profiles.json
    
    Returns:
        Formality level (0.0 to 1.0), where higher = more formal.
    """
    profiles = _load_city_profiles(config_path)
    
    if city_name not in profiles:
        return 0.5  # Default moderate formality
    
    profile = profiles[city_name]
    lexical_style = profile.get('lexical_style', {})
    return lexical_style.get('formality_level', 0.5)


def get_city_phrases(city_name: str, config_path: Optional[str] = None) -> List[str]:
    """
    Get the list of phrases for a city.
    
    Args:
        city_name: Name of the city
        config_path: Optional path to city_profiles.json
    
    Returns:
        List of phrase strings.
    """
    profiles = _load_city_profiles(config_path)
    
    if city_name not in profiles:
        return []
    
    profile = profiles[city_name]
    lexical_style = profile.get('lexical_style', {})
    return lexical_style.get('phrases', [])
