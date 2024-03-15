"""
Modified from https://github.com/coqui-ai/TTS
"""

import numpy as np
from TTS.config import load_config
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak
from data_helper.cleaners import text_cleaners

def encode(text: str, characters):
    """Encodes a string of text as a sequence of IDs."""
    token_ids = []
    for char in text:
        try:
            idx = characters.char_to_id(char)
            token_ids.append(idx)
        except KeyError:
            # discard but store not found characters
            # if char not in self.not_found_characters:
            #     self.not_found_characters.append(char)
            print(text)
            print(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")
    return token_ids

class Phonemizer:
    def __init__(self, language):
        tts_config_path = "./data_helper/config.json"
        config = load_config(tts_config_path)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        characters = tokenizer.characters
        characters.vocab = characters.vocab[:-1] + ['̃', '̊', '̝', '-', '̪', '^']

        if language=="fr":
            espeak_language = "fr-fr"
        elif language=="pt":
            espeak_language = "pt-pt"
        elif language=="ar":
            espeak_language = "ar-ar"
        else:
            espeak_language = language

        self.language = language
        self.characters = characters
        self.phonemizer = ESpeak(language=espeak_language)

    def text2phoneme_unit(self, text):
        text = text_cleaners(text, lang=self.language)

        phoneme = self.phonemizer.phonemize(text, separator="")

        phoneme_unit = encode(phoneme, characters=self.characters)
        phoneme_unit = np.asarray(phoneme_unit, dtype=np.int64)

        return phoneme_unit