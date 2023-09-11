import argparse
import tqdm
import torch

from fairseq import utils
from fairseq_cli.generate import get_symbols_to_strip_from_output

from speech2unit.inference import load_model as load_speech2unit_model
from unit2unit.inference import load_model as load_unit2unit_model
from unit2speech.inference import load_model as load_unit2speech_model

from util import process_units, save_speech

class SpeechToSpeechPipeline:
    def __init__(self, 
        hubert_reader, kmeans_model,
        task, generator,
        vocoder,
        use_cuda=False
    ):
        self.hubert_reader = hubert_reader
        self.kmeans_model = kmeans_model
        self.task = task
        self.generator = generator
        self.vocoder = vocoder
        self.use_cuda = use_cuda

    def process_speech2unit(self, speech_path):
        feats = self.hubert_reader.get_feats(speech_path)
        feats = feats.cpu().numpy()

        pred = self.kmeans_model.predict(feats)
        pred_str = " ".join(str(p) for p in pred)

        return pred_str

    def process_unit2unit(self, unit):
        unit = list(map(int, unit.strip().split()))
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

    hubert_reader, kmeans_model = load_speech2unit_model(args.mhubert_path, args.kmeans_path, use_cuda=use_cuda)
    task, generator = load_unit2unit_model(args.utut_path, args.src_lang, args.tgt_lang, use_cuda=use_cuda)
    vocoder = load_unit2speech_model(args.vocoder_path, args.vocoder_cfg_path, use_cuda=use_cuda)

    pipeline = SpeechToSpeechPipeline(
        hubert_reader, kmeans_model,
        task, generator,
        vocoder,
        use_cuda=use_cuda
    )

    for in_wav_path, out_wav_path in tqdm.tqdm(
        zip(args.in_wav_path, args.out_wav_path),
        total=min(len(args.in_wav_path), len(args.out_wav_path))
    ):
        src_unit = pipeline.process_speech2unit(in_wav_path)
        tgt_unit = pipeline.process_unit2unit(src_unit)
        tgt_speech = pipeline.process_unit2speech(tgt_unit)

        save_speech(tgt_speech.detach().cpu().numpy(), out_wav_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-wav-path", type=str, required=True, nargs="*", help="File path of source speech input"
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
        "--mhubert-path",
        type=str,
        required=True,
        help="Pretrained mHuBERT model checkpoint"
    )
    parser.add_argument(
        "--kmeans-path",
        type=str,
        required=True,
        help="K-means model file path to use for inference",
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