from random import random
from PIL import Image
from typing import Any
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from shared import TrainingObject

from pair_provider import PairProvider
from loss_meter import LossMeter
from shared import TrainingPair, DataLoaderType


class DreamBoothDataset(Dataset, TrainingObject):
    """
    A dataset to prepare the instance and class images with the prompts for
    fine-tuning the model. It pre-processes the images and the tokenizer
    prompts.
    """

    def __init__(
        self,
        inst_img_prompt_tuples,
        class_img_prompt_tuples,
        tokenizer
    ):
        self.loss_dict = dict()
        self.tokenizer = tokenizer

        # print(len(inst_img_prompt_tuples))
        self.inst_img_prompt_tuples = inst_img_prompt_tuples
        self.class_img_prompt_tuples = class_img_prompt_tuples
        self.class_images_randomizer_stack = []

        self.num_inst_images = len(self.inst_img_prompt_tuples)
        self.num_class_images = len(self.class_img_prompt_tuples)

        self.image_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5 * self.args.hflip),
                (transforms.Resize(
                    self.args.resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR)),
                (transforms.CenterCrop(self.args.resolution)
                    if self.args.center_crop
                    else transforms.RandomCrop(self.args.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._get_length()

    def __getitem__(self, index):
        # print('getitem of DreamBoothDataset')
        return self._internal_get_item(index,
                                       self._get_random_class_image_index())

    def _internal_get_item(self, instance_index: int, class_index: int) -> Any:
        print(f'dreamboothdataset_internal_getitem_ {instance_index}, {class_index}')
        # print('_internal_get_item of DreamBoothDataset')
        data_set_item = {}
        instance_path, instance_prompt = \
            self.inst_img_prompt_tuples[instance_index]  # % self.num_inst_images]

        data_set_item["instance_images"] = self._transform_image(instance_path)
        data_set_item["instance_prompt_ids"] = \
            self._get_input_ids_from_tokenizer(instance_prompt)

        if self.args.with_prior_preservation:
            class_path, class_prompt = self.class_img_prompt_tuples[class_index]

            data_set_item["class_images"] = self._transform_image(class_path)
            data_set_item["class_prompt_ids"] = \
                self._get_input_ids_from_tokenizer(class_prompt)

        return data_set_item

    def _get_length(self) -> int:
        return max(self.num_class_images, self.num_inst_images)

    def _transform_image(self, image_path: str) -> Any:
        img = Image.open(image_path)
        if not img.mode == 'RGB':
            img = img.convert('RGB')

        result = self.image_transformer(img)
        img.close()

        return result

    def _get_input_ids_from_tokenizer(self, prompt: str) -> Any:
        return self.tokenizer(
            prompt,
            padding=("max_length"
                     if self.args.pad_tokens
                     else "do_not_pad"),
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

    def _get_random_class_image_index(self) -> int:
        if len(self.class_images_randomizer_stack) == 0:
            self.class_images_randomizer_stack = \
                [x for x in range(self.num_class_images)]

        random_index = random.randint(
            0,
            len(self.class_images_randomizer_stack) - 1)

        result = self.class_images_randomizer_stack.pop(random_index)

        return result

    # def _cache_latents(train_dataset: DreamBoothDataset,
    #                   train_dataloader,
    #                   vae
    #                   ) -> Tuple[DreamBoothDataset, Any]:
    #     latents_cache = []
    #     text_encoder_cache = []
    #     for batch in tqdm(train_dataloader, desc="Caching latents"):
    #         with torch.no_grad():
    #             batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
    #             batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
    #             latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
    #             if args.train_text_encoder:
    #                 text_encoder_cache.append(batch["input_ids"])
    #             else:
    #                 text_encoder_cache.append(text_encoder(batch["input_ids"])[0])
    #     del train_dataset
    #     train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
    #     del train_dataloader
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

    #     del vae
    #     if not args.train_text_encoder:
    #         del text_encoder
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     return train_dataset, train_dataloader


class SmartCrossProductDataSet(DreamBoothDataset):
    def __init__(self,
                 pair_provider: PairProvider,
                 #  pairs: list[TrainingPair],
                 loss_meter: LossMeter,
                 #  loss_dict: dict[tuple[int, int], list[float]],
                 create_dataloader_fn,
                 accelerator,
                 text_encoder,
                 weight_dtype,
                 vae,
                 *args):
        super().__init__(*args)
        self.pair_provider = pair_provider
        # self.pairs = pairs
        self.loss_meter = loss_meter
        # self.loss_dict = loss_dict
        self.accelerator = accelerator
        self.text_encoder = text_encoder
        self.weight_dtype = weight_dtype
        self.vae = vae
        # print(type(vae))
        self._internal_dataloader = \
            create_dataloader_fn(type=DataLoaderType.REGULAR,
                                 dataset=self
                                 )
        self._cache = dict()
        self._rebuilding_cache = False
        self.text_encoder_cache = []
        # self.pair_index = 0
        # self.last_pair = (0, 0)
        self.highest_loss_pairs = []
        self.cache_pairs = []

    def __getitem__(self, index) -> dict:
        # print('getitem of SmartCrossProductDataSet')
        if self._rebuilding_cache:
            if index <= self.num_class_images / 2 and self.pair_index > 0:
                print('now recaching highest loss pairs')
                self.cache_pairs.append(self.highest_loss_pairs[index])
                return super()._internal_get_item(*self.highest_loss_pairs[index])
            else:
                self.cache_pairs.append(self.pairs[self.pair_index + index])
                return super()._internal_get_item(*self.pairs[self.pair_index + index])

        # print(f'last pair {self.last_pair}')
        return self._internal_get_item(*self.last_pair)

    def _get_length(self) -> int:
        return self.num_inst_images

    def _internal_get_item(self, instance_index: int, class_index: int) -> Any:
        print('_internal_get_item of SmartCrossProductDataSet')
        if len(self._cache) > 0:
            print('cache not empty')
            if len(self.highest_loss_pairs) > 0:
                print(f'using {self.last_pair} from highest loss pairs as last pair')
                self.last_pair = self.highest_loss_pairs.pop()
            else:
                self.last_pair = self.pairs[self.pair_index]
                self.pair_index += 1

            pair_key = TrainingPair(instance_index, class_index)
            print(f'pair_key {pair_key}')
            print(self._cache.keys())
            latent, text_enc_cache = self._cache[pair_key]
            # del self._cache[pair_key]
            return latent, text_enc_cache
        else:
            print('rebuild cache')
            self._rebuilding_cache = True
            # if self.pair_index > 0:
            #     self.highest_loss_pairs = \
            #         sorted(self.loss_dict,
            #                key=self.loss_dict.get,
            #                reverse=True
            #                )[:round(self.num_inst_images / 2)]
            for batch in tqdm(self._internal_dataloader, desc="Rebuilding cache..."):
                with torch.no_grad():
                    batch["pixel_values"] = \
                        batch["pixel_values"].to(
                            self.accelerator.device,
                            non_blocking=True,
                            dtype=self.weight_dtype)
                    batch["input_ids"] = \
                        batch["input_ids"].to(
                            self.accelerator.device,
                            non_blocking=True)
                    latent = self.vae.encode(batch["pixel_values"]).latent_dist
                    # if not args.train_text_encoder:
                    #  self.text_encoder_cache.append(batch["input_ids"])
                    # else:
                    text_enc_cache = batch["input_ids"]
                    # text_enc_cache = self.text_encoder(batch["input_ids"])[0]
                self._cache[self.cache_pairs.pop(0)] = (latent, text_enc_cache)
            # pairs_to_be_cached = self.pairs[self.pair_index - 1:
            #                                 self.pair_index + self.num_inst_images - 1]

            # for pair in pairs_to_be_cached:
            #     pass
            print(f'dict keys after caching {self._cache.keys()}')
            self._rebuilding_cache = False
            pair_key = next(iter(self._cache))
            latent, text_enc_cache = self._cache[pair_key]
            # del self._cache[pair_key]
            return latent, text_enc_cache

            # print(pairs_to_be_cached)
            # print(f'len pairs_to_be_cached {pairs_to_be_cached}')

    def _cache_latents(self, pairs: list) -> dict:
        pass

    def add_loss_for_last_pair(self, loss: float) -> None:
        self.loss_dict[self.last_pair].append(loss)


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        # traceback.print_stack()
        return self.latents_cache[index], self.text_encoder_cache[index]
