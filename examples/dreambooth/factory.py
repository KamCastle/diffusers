from pathlib import Path
from typing import Any
import torch
from datasets import DreamBoothDataset, \
    SmartCrossProductDataSet, LossMeter, DataLoaderType
from examples.dreambooth.pair_provider import PairProvider
from examples.dreambooth.parsers import ConceptsListParser
from shared import TrainingObject
from shared import TrainingPair


class DreamBoothFactory(TrainingObject):
    def __init__(
        self,
        pair_provider: PairProvider,
        collate_fn,
        concepts_list,
        tokenizer,
        vae,
        accelerator,
        text_encoder,
        weight_dtype,
        with_prior_preservation,
        use_smart_cross_products,
        resolution,
        train_batch_size,
        center_crop,
        num_class_images,
        pad_tokens,
        hflip
    ) -> None:
        # print('Building DreamBoothFactory')
        self.pair_provider = pair_provider
        self.collate_fn = collate_fn
        self.with_prior_preservation = with_prior_preservation
        self.vae = vae
        self.accelerator = accelerator
        self.text_encoder = text_encoder
        self.weight_dtype = weight_dtype
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.train_batch_size = train_batch_size
        self.center_crop = center_crop
        self.inst_img_prompt_tuples = []
        self.class_img_prompt_tuples = []
        self.pad_tokens = pad_tokens
        self.hflip = hflip
        self.use_smart_cross_products = use_smart_cross_products

        clp = ConceptsListParser()
        self.inst_img_prompt_tuples, self.class_img_prompt_tuples = \
            clp.parse(concepts_list, num_class_images, with_prior_preservation)

    def create_dataset(self) -> DreamBoothDataset:
        dataset_args = \
            [
                self.inst_img_prompt_tuples,
                self.class_img_prompt_tuples,
                self.tokenizer,
                self.with_prior_preservation,
                self.resolution,
                self.center_crop,
                self.pad_tokens,
                self.hflip
            ]

        if self.use_smart_cross_products:
            # print('Creating SmartCrossProductDataSet')
            self.loss_meter = LossMeter(
                {pair: [0] for pair in self.pair_provider.get_all_pairs()})
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
        # print('creating data loader')
        if dataset is None:
            dataset = self.create_dataset()

        if type == DataLoaderType.REGULAR:
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     self.train_batch_size,
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
