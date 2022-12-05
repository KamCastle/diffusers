from pathlib import Path
from typing import Any
import torch
from datasets import DreamBoothDataset, \
    SmartCrossProductDataSet, LossMeter, DataLoaderType
from pair_provider import PairProvider
from parsers import ConceptsListParser
from shared import TrainingObject
from shared import TrainingPair


class DreamBoothFactory(TrainingObject):
    def __init__(
        self,
        pair_provider: PairProvider,
        loss_meter: LossMeter,
        collate_fn,
        tokenizer,
        vae,
        accelerator,
        text_encoder,
        weight_dtype
    ) -> None:
        # print('Building DreamBoothFactory')
        self.pair_provider = pair_provider
        self.loss_meter = loss_meter
        self.collate_fn = collate_fn
        self.vae = vae
        self.accelerator = accelerator
        self.text_encoder = text_encoder
        self.weight_dtype = weight_dtype
        self.tokenizer = tokenizer
        self.inst_img_prompt_tuples = []
        self.class_img_prompt_tuples = []

        clp = ConceptsListParser()
        self.inst_img_prompt_tuples, self.class_img_prompt_tuples = \
            clp.parse(self.args.concepts_list,
                      self.args.num_class_images,
                      self.args.with_prior_preservation)

    def create_dataset(self) -> DreamBoothDataset:
        dataset_args = \
            [
                self.inst_img_prompt_tuples,
                self.class_img_prompt_tuples,
                self.tokenizer
            ]

        if self.args.use_smart_cross_products:
            return SmartCrossProductDataSet(self.pair_provider,
                                            self.loss_meter,
                                            self.create_dataloader,
                                            self.accelerator,
                                            self.text_encoder,
                                            self.weight_dtype,
                                            self.vae,
                                            *dataset_args)
        else:
            return DreamBoothDataset(*dataset_args)

    def create_dataloader(self,
                          type: DataLoaderType,
                          dataset: DreamBoothDataset = None
                          ) -> Any:
        if dataset is None:
            dataset = self.create_dataset()

        if type == DataLoaderType.REGULAR:
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     self.args.train_batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate_fn,
                                                     pin_memory=True
                                                     )
        else:
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     collate_fn=lambda x: x,
                                                     )

        return dataloader

    def _build_cross_product_pairs(self) -> list[TrainingPair]:
        result = [TrainingPair(instance_index, class_index)
                  for instance_index in range(len(self.inst_img_prompt_tuples))
                  for class_index in range(len(self.class_img_prompt_tuples))]

        return result
