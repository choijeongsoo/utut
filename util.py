import os
import soundfile as sf

def process_units(units, reduce=False):
    if not reduce:
        return units

    out = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
    return out

def save_unit(unit, unit_path):
    os.makedirs(os.path.dirname(unit_path), exist_ok=True)
    with open(unit_path, "w") as f:
        f.write(unit)

def save_speech(speech, speech_path, sampling_rate=16000):
    os.makedirs(os.path.dirname(speech_path), exist_ok=True)
    sf.write(
        speech_path,
        speech,
        sampling_rate,
    )