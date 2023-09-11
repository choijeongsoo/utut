# This code is from https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/generate_waveform_from_code.py

import argparse
import tqdm
import json
import torch

from fairseq import utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

from util import save_speech

def load_model(model_path, cfg_path, use_cuda=False):
    with open(cfg_path) as f:
        vocoder_cfg_path = json.load(f)
    vocoder = CodeHiFiGANVocoder(model_path, vocoder_cfg_path)
    if use_cuda:
        vocoder = vocoder.cuda()
    return vocoder

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    vocoder = load_model(args.vocoder_path, args.vocoder_cfg_path, use_cuda=use_cuda)

    for in_unit_path, out_wav_path in tqdm.tqdm(
        zip(args.in_unit_path, args.out_wav_path),
        total=min(len(args.in_unit_path), len(args.out_wav_path))
    ):
        with open(in_unit_path) as f:
            unit = list(map(int, f.readline().strip().split()))

        sample = {
            "code": torch.LongTensor(unit).view(1,-1),
        }
        sample = utils.move_to_cuda(sample) if use_cuda else sample

        wav = vocoder(sample, dur_prediction=True)

        save_speech(wav.detach().cpu().numpy(), out_wav_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-unit-path", type=str, required=True, nargs="*", help="File path of unit input"
    )
    parser.add_argument(
        "--out-wav-path", type=str, required=True, nargs="*", help="File path of speech output"
    )
    parser.add_argument(
        "--vocoder-path", type=str, required=True, help="path to the CodeHiFiGAN vocoder"
    )
    parser.add_argument(
        "--vocoder-cfg-path",
        type=str,
        required=True,
        help="path to the CodeHiFiGAN vocoder config",
    )
    parser.add_argument(
        "--dur-prediction",
        action="store_true",
        help="enable duration prediction (for reduced/unique code sequences)",
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()