# UTUT

Official PyTorch implementation for the following paper:
> **Textless Unit-to-Unit Pre-training for Many-to-Many Multimodal-to-Speech Machine Translation by Learning Unified Speech and Text Representations**<br>
> [Minsu Kim](https://sites.google.com/view/ms-dot-k)\*, [Jeongsoo Choi](https://choijeongsoo.github.io)\*, [Dahun Kim](https://mcahny.github.io), [Yong Man Ro](https://www.ivllab.kaist.ac.kr/people/professor)<br>
> \[[Demo](https://choijeongsoo.github.io/utut)\]

<div align="center"><img width="80%" src="imgs/fig1.png?raw=true"/></div>


## Setup
Python >=3.7,<3.11
```
git clone -b main --single-branch https://github.com/choijeongsoo/utut
cd utut
git submodule init
git submodule update
pip install -e fairseq
pip install -r requirements.txt
apt-get install espeak
```

## Model Checkpoints

### Speech to Unit Quantization

- mHuBERT Base, layer 11, km 1000

  > reference: [textless_s2st_real_data](https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_to_speech/docs/textless_s2st_real_data.md#hubert)

### Unit to Unit Translation (UTUT)

- Pre-trained Model

  Task | Pretraining Data | Model
  |---|---|---
  STS | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) (from year 2013), [mTEDx](https://www.openslr.org/100) | [download](https://drive.google.com/file/d/1MEETNogbSgmqkhvzIO4SRSbh0o0dV-Sr/view?usp=sharing)
  TTS | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) (from year 2013), [mTEDx](https://www.openslr.org/100) | [download](https://drive.google.com/file/d/17CqXLMftL0BBaa_GAbLLzYPDTDRzpbma/view?usp=sharing)
  TTST | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) (from year 2013), [mTEDx](https://www.openslr.org/100) | [download](https://drive.google.com/file/d/1etg6FMqIh3uaimaVpEuLVvBA8Rluh57Q/view?usp=sharing)

<!-- Current versiocn only provides pre-trained UTUT model checkpoint and inference code for multilingual speech-to-speech translation. -->

### Unit to Speech Synthesis

- En (English), Es (Spanish), and Fr (French)

  > reference: [textless_s2st_real_data](https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_to_speech/docs/textless_s2st_real_data.md#unit-based-hifi-gan-vocoder)

- It (Italian), De (German), and Nl (Dutch)

  Unit config | Unit size | Vocoder language | Dataset | Model
  |---|---|---|---|---
  mHuBERT, layer 11 | 1000 | It | [M-AILABS](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset) (male) | [ckpt](https://drive.google.com/file/d/1--j3LGnxmEdyb-urT0hilbXsOX66B0g_/view?usp=sharing), [config](https://drive.google.com/file/d/1bkHMa4ZG5OqH5_TLZJguFAa-VButFBxb/view?usp=sharing)
  mHuBERT, layer 11 | 1000 | De | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://drive.google.com/file/d/1l7s9NWjc8vRQnVGspL7c13cQD6PHIZLy/view?usp=sharing), [config](https://drive.google.com/file/d/1n55q_-9Ea72BDVIUPCH4ynjXAZaEjnv8/view?usp=sharing)
  mHuBERT, layer 11 | 1000 | Nl | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://drive.google.com/file/d/1j2DI--8S-qxn_10_p2kQOsvIlO2ArjsO/view?usp=sharing), [config](https://drive.google.com/file/d/188mZhUeuyF1fNcN7pnx3oqhbim5LJ_fC/view?usp=sharing)


## Inference
UTUT is pre-trained on Voxpopuli and mTEDx, where a large portion of data is from European Parliament events. <br>
Before utilizing the pre-trained model, please consider the data domain where you want to apply it.

### Pipeline for Speech-to-Speech Translation (STS)
```
$ cd utut
$ PYTHONPATH=fairseq python inference_sts.py \
  --in-wav-path samples/en/1.wav samples/en/2.wav samples/en/3.wav \
  --out-wav-path samples/es/1.wav samples/es/2.wav samples/es/3.wav \
  --src-lang en --tgt-lang es \
  --mhubert-path /path/to/mhubert_base_vp_en_es_fr_it3.pt \
  --kmeans-path /path/to/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin \
  --utut-path /path/to/utut_sts.pt \
  --vocoder-path /path/to/vocoder_es.pt \
  --vocoder-cfg-path /path/to/config_es.json
```

### Pipeline for Text-to-Speech Synthesis (TTS)
```
$ cd utut
$ PYTHONPATH=fairseq python inference_tts.py \
  --in-txt-path samples/en/a.txt samples/en/b.txt samples/en/c.txt \
  --out-wav-path samples/en/a.wav samples/en/b.wav samples/en/c.wav \
  --src-lang en --tgt-lang en \
  --utut-path /path/to/utut_tts.pt \
  --vocoder-path /path/to/vocoder_en.pt \
  --vocoder-cfg-path /path/to/config_en.json
```

### Pipeline for Text-to-Speech Translation (TTST)
```
$ cd utut
$ PYTHONPATH=fairseq python inference_tts.py \
  --in-txt-path samples/en/a.txt samples/en/b.txt samples/en/c.txt \
  --out-wav-path samples/es/a.wav samples/es/b.wav samples/es/c.wav \
  --src-lang en --tgt-lang es \
  --utut-path /path/to/utut_ttst.pt \
  --vocoder-path /path/to/vocoder_es.pt \
  --vocoder-cfg-path /path/to/config_es.json
```

19 source languages: en (English), es (Spanish), fr (French), it (Italian), pt (Portuguese), el (Greek), ru (Russian), cs (Czech), da (Danish), de (German), fi (Finnish), hr (Croatian), hu (Hungarian), lt (Lithuanian), nl (Dutch), pl (Polish), ro (Romanian), sk (Slovak), and sl (Slovene)

6 target languages: en (English), es (Spanish), fr (French), it (Italian), de (German), and nl (Dutch)

<!-- 
### Load a UTUT pre-trained model
```
$ cd utut
$ python
>>> import fairseq
>>> import unit2unit.utut_pretraining
>>> ckpt_path = "/path/to/utut_sts.pt"
>>> models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
>>> model = models[0]
```

### Speech to Unit
```
$ cd utut
$ PYTHONPATH=fairseq python -m speech2unit.inference \
  --in-wav-path samples/en/1.wav samples/en/2.wav samples/en/3.wav \
  --out-unit-path samples/en/1.unit samples/en/2.unit samples/en/3.unit \
  --mhubert-path /path/to/mhubert_base_vp_en_es_fr_it3.pt \
  --kmeans-path /path/to/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin \
```

We use mHuBERT model trained on 3 languages (en, es, and fr) as the quantizer.

### Unit to Unit
```
$ cd utut
$ python -m unit2unit.inference \
  --in-unit-path samples/en/1.unit samples/en/2.unit samples/en/3.unit \
  --out-unit-path samples/es/1.unit samples/es/2.unit samples/es/3.unit \
  --utut-path /path/to/utut_sts.pt \
  --src-lang en --tgt-lang es
```

UTUT supports 19 languages: en (English), es (Spanish), fr (French), it (Italian), pt (Portuguese), el (Greek), ru (Russian), cs (Czech), da (Danish), de (German), fi (Finnish), hr (Croatian), hu (Hungarian), lt (Lithuanian), nl (Dutch), pl (Polish), ro (Romanian), sk (Slovak), and sl (Slovene)

### Unit to Speech
```
$ cd utut
$ python -m unit2speech.inference \
  --in-unit-path samples/es/1.unit samples/es/2.unit samples/es/3.unit \
  --out-wav-path samples/es/1.wav samples/es/2.wav samples/es/3.wav \
  --vocoder-path /path/to/vocoder_es.pt \
  --vocoder-cfg-path /path/to/config_es.json
```

We support 6 languages: en (English), es (Spanish), fr (French), it (Italian), de (German), and nl (Dutch)
 -->

## Acknowledgement

This repository is built upon [Fairseq](https://github.com/pytorch/fairseq) and [speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis). We appreciate the open source of the projects.


## Citation

If our work is useful for your research, please cite the following paper:
```bibtex
@article{kim2023many,
    title={Many-to-Many Spoken Language Translation via Unified Speech and Text Representation Learning with Unit-to-Unit Translation},
    author={Minsu Kim and Jeongsoo Choi and Dahun Kim and Yong Man Ro},
    journal={arXiv preprint arXiv:2308.01831},
    year={2023}
}
