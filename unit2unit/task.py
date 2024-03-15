import os
import logging
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_denoising import MultilingualDenoisingConfig, MultilingualDenoisingTask
from fairseq.data import Dictionary

logger = logging.getLogger(__name__)

@register_task("utut_pretraining", dataclass=MultilingualDenoisingConfig)
class UTUTPretrainingTask(MultilingualDenoisingTask):
    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        lang_list = self.cfg.langs.split(",")

        lang_token_ids = {
            self.dictionary.index("[{}]".format(lang))
            for lang in lang_list
        }

        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}

        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids

        extra_gen_cls_kwargs["eos"] = self.dictionary.index("[{}]".format(self.target_language))

        extra_gen_cls_kwargs["tokens_to_suppress"] = [
            "[{}]".format(lang) for lang in lang_list if lang != self.target_language
        ] + [self.dictionary[self.mask_idx]]

        return super().build_generator(
            models,
            args,
            seq_gen_cls=seq_gen_cls,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs,
        )

@register_task("utut_ttst", dataclass=MultilingualDenoisingConfig)
class UTUTTTSTTask(UTUTPretrainingTask):
    @classmethod
    def setup_task(cls, cfg: MultilingualDenoisingConfig, **kwargs):
        """Setup the task."""
        paths = cfg.data.split(":")
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        phoneme_dictionary = Dictionary.load(os.path.join(paths[0], "phoneme_dict.txt"))

        data_path = paths[0]
        if cfg.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = cfg.langs.split(",")

        if cfg.add_lang_token:
            for lang in languages:
                dictionary.add_symbol("[{}]".format(lang))
                phoneme_dictionary.add_symbol("[{}]".format(lang))

        logger.info("dictionary: {} types".format(len(dictionary)))
        if not hasattr(cfg, "shuffle_instance"):
            cfg.shuffle_instance = False
        return cls(cfg, dictionary, phoneme_dictionary)

    def __init__(self, cfg, dictionary, phoneme_dictionary):
        super().__init__(cfg, dictionary)
        self.phoneme_dictionary = phoneme_dictionary
        self.phoneme_dictionary.add_symbol("<mask>")

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.phoneme_dictionary
