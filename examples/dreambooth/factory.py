from pathlib import Path
from typing import Any
import torch
from datasets import DreamBoothDataset, \
    SmartCrossProductDataSet, LossMeter, DataLoaderType
from shared import TrainingPair


class DreamBoothFactory:
    def __init__(
        self,
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

        for concept in concepts_list:
            concept_inst_tuples = [(x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()]
            self.inst_img_prompt_tuples.extend(concept_inst_tuples)

            if with_prior_preservation:
                concept_class_tuples = [(x, concept["class_prompt"]) for x in Path(concept["class_data_dir"]).iterdir() if x.is_file()]
                self.class_img_prompt_tuples.extend(concept_class_tuples[:num_class_images])

    def create_dataset(self) -> DreamBoothDataset:
        args = \
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
            self.pairs = self._build_cross_product_pairs()
            self.loss_dict = LossMeter({pair: [0] for pair in self.pairs})
            return SmartCrossProductDataSet(self.pairs,
                                            self.loss_dict,
                                            self.create_dataloader,
                                            self.accelerator,
                                            self.text_encoder,
                                            self.weight_dtype,
                                            self.vae,
                                            *args)
        else:
            return DreamBoothDataset(*args)

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