import argparse
import tqdm
import torch

from fairseq import utils
from fairseq_cli.generate import get_symbols_to_strip_from_output

from phonemize import Phonemizer
from unit2unit.inference import load_model as load_unit2unit_model
from unit2speech.inference import load_model as load_unit2speech_model

from util import process_units, save_speech

class TextToSpeechPipeline:
    def __init__(self,
        task, generator,
        vocoder,
        use_cuda=False
    ):
        self.task = task
        self.generator = generator
        self.vocoder = vocoder
        self.use_cuda = use_cuda

    def process_unit2unit(self, unit):
        # unit = list(map(int, unit.strip().split()))
        unit = self.task.source_dictionary.encode_line(
            " ".join(map(lambda x: str(x), process_units(unit, reduce=True))),
            add_if_not_exist=False,
            append_eos=True,
        ).long()
        unit = torch.cat([
            unit.new([self.task.source_dictionary.bos()]),
            unit,
            unit.new([self.task.source_dictionary.index("[{}]".format(self.task.source_language))])
        ])

        sample = {"net_input": {
            "src_tokens": torch.LongTensor(unit).view(1,-1),
        }}
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        pred = self.task.inference_step(
            self.generator,
            None,
            sample,
        )[0][0]

        pred_str = self.task.target_dictionary.string(
            pred["tokens"].int().cpu(),
            extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator)
        )

        return pred_str

    def process_unit2speech(self, unit):
        unit = list(map(int, unit.strip().split()))

        sample = {
            "code": torch.LongTensor(unit).view(1,-1),
        }
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        wav = self.vocoder(sample, True)

        return wav

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    phonemizer = Phonemizer(args.src_lang)
    task, generator = load_unit2unit_model(args.utut_path, args.src_lang, args.tgt_lang, use_cuda=use_cuda)
    vocoder = load_unit2speech_model(args.vocoder_path, args.vocoder_cfg_path, use_cuda=use_cuda)

    pipeline = TextToSpeechPipeline(
        task, generator,
        vocoder,
        use_cuda=use_cuda
    )

    for in_txt_path, out_wav_path in tqdm.tqdm(
        zip(args.in_txt_path, args.out_wav_path),
        total=min(len(args.in_txt_path), len(args.out_wav_path))
    ):
        src_text = open(in_txt_path).readline().strip()
        src_unit = phonemizer.text2phoneme_unit(src_text)
        tgt_unit = pipeline.process_unit2unit(src_unit)
        tgt_speech = pipeline.process_unit2speech(tgt_unit)

        save_speech(tgt_speech.detach().cpu().numpy(), out_wav_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-txt-path", type=str, required=True, nargs="*", help="File path of source text input"
    )
    parser.add_argument(
        "--out-wav-path", type=str, required=True, nargs="*", help="File path of translated speech output"
    )
    parser.add_argument(
        "--src-lang", type=str, required=True,
        choices=["en","es","fr","it","pt","el","ru","cs","da","de","fi","hr","hu","lt","nl","pl","ro","sk","sl"],
        help="source language"
    )
    parser.add_argument(
        "--tgt-lang", type=str, required=True,
        choices=["en","es","fr","it","pt","el","ru","cs","da","de","fi","hr","hu","lt","nl","pl","ro","sk","sl"],
        help="target language"
    )
    parser.add_argument(
        "--utut-path", type=str, required=True, help="path to the UTUT pre-trained model"
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
    parser.add_argument("--cpu", action="store_true", help="run on CPU")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()