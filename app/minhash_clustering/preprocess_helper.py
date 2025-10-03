import re
import unicodedata
from typing import Dict

CLEAN_PATTERN = re.compile(r'[^\w\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')

def preprocess_text(text: str, options: Dict[str, bool] = None) -> str:
    if options is None:
        options = {
            'lowercase': False,
            'remove_diacritics': False,
            'remove_punctuation': False
        }

    text = str(text)

    if options.get('lowercase', True):
        text = text.lower()

    if options.get('remove_diacritics', True):
        text = unicodedata.normalize('NFD', text)
        text = ''.join(
            char for char in text
            if unicodedata.category(char) != 'Mn'
        )
        text = unicodedata.normalize('NFC', text)
    else:
        text = unicodedata.normalize('NFC', text)

    if options.get('remove_punctuation', True):
        text = CLEAN_PATTERN.sub(' ', text)

    text = WHITESPACE_PATTERN.sub(' ', text)

    return text.strip()