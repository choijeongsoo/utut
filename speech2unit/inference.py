# This code is from https://github.com/facebookresearch/fairseq/blob/main/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py

import argparse
import tqdm
import joblib
import torch

from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import (
    HubertFeatureReader,
)

from util import save_unit

def load_model(model_path, kmeans_path, use_cuda=False):
    hubert_reader = HubertFeatureReader(
        checkpoint_path=model_path,
        layer=11,
        use_cuda=use_cuda,
    )
    kmeans_model = joblib.load(open(kmeans_path, "rb"))
    kmeans_model.verbose = False

    return hubert_reader, kmeans_model

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    hubert_reader, kmeans_model = load_model(args.mhubert_path, args.kmeans_path, use_cuda=use_cuda)

    for in_wav_path, out_unit_path in tqdm.tqdm(
        zip(args.in_wav_path, args.out_unit_path),
        total=min(len(args.in_wav_path), len(args.out_unit_path))
    ):
        feats = hubert_reader.get_feats(in_wav_path)
        feats = feats.cpu().numpy()

        pred = kmeans_model.predict(feats)
        pred_str = " ".join(str(p) for p in pred)

        save_unit(pred_str, out_unit_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-wav-path", type=str, required=True, nargs="*", help="File path of speech input"
    )
    parser.add_argument(
        "--out-unit-path", type=str, required=True, nargs="*", help="File path of unit output"
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
    parser.add_argument("--cpu", action="store_true", help="run on CPU")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()