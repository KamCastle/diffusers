import argparse
from enum import Enum, auto
import hashlib
import itertools
import random
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from shared import TrainingPair


torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--save_sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--save_sample_negative_prompt",
        type=str,
        default=None,
        help="The negative prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=50,
        help="The number of inference steps for save sample.",
    )
    parser.add_argument(
        "--pad_tokens",
        default=False,
        action="store_true",
        help="Flag to pad tokens to length 77.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--shuffle_after_epoch",
        action="store_true",
        help="Whether or not to shuffle and recache training dataset after every epoch."
    )
    parser.add_argument(
        "--use_smart_cross_products",
        action="store_true",
        help="Whether or not to use a smart loss-based cross product dataset for training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_interval", type=int, default=10_000, help="Save weights every N steps.")
    parser.add_argument("--save_min_steps", type=int, default=0, help="Start saving weights after N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--not_cache_latents", action="store_true", help="Do not precompute and cache latents from VAE.")
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class ImageType(Enum):
    CLASS = auto()
    INSTANCE = auto()


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizer prompts.
    """

    def __init__(
        self,
        inst_img_prompt_tuples,
        class_img_prompt_tuples,
        tokenizer,
        with_prior_preservation=True,
        resolution=512,
        center_crop=False,
        pad_tokens=False,
        hflip=False
    ):
        self.loss_dict = dict()
        self.center_crop = center_crop
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.pad_tokens = pad_tokens

        # print(len(inst_img_prompt_tuples))
        self.inst_img_prompt_tuples = inst_img_prompt_tuples
        self.class_img_prompt_tuples = class_img_prompt_tuples
        self.class_images_randomizer_stack = []

        self.num_inst_images = len(self.inst_img_prompt_tuples)
        self.num_class_images = len(self.class_img_prompt_tuples)

        self.image_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5 * hflip),
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(resolution),
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

    def add_loss_for_last_pair(self, loss: float) -> None:
        pass

    def _internal_get_item(self, instance_index: int, class_index: int) -> Any:
        print(f'dreamboothdataset_internal_getitem_ {instance_index}, {class_index}')
        # print('_internal_get_item of DreamBoothDataset')
        data_set_item = {}
        instance_path, instance_prompt = \
            self.inst_img_prompt_tuples[instance_index]  # % self.num_inst_images]

        data_set_item["instance_images"] = self._transform_image(instance_path)
        data_set_item["instance_prompt_ids"] = \
            self._get_input_ids_from_tokenizer(instance_prompt)

        if self.with_prior_preservation:
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
            padding="max_length" if self.pad_tokens else "do_not_pad",
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
                 pairs: list[tuple[int, int]],
                 loss_dict: dict[tuple[int, int], list[float]],
                 create_dataloader_fn,
                 accelerator,
                 text_encoder,
                 weight_dtype,
                 vae,
                 *args):
        super().__init__(*args)
        self.pairs = pairs
        self.loss_dict = loss_dict
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
        self.pair_index = 0
        self.last_pair = (0, 0)
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

            
            pair_key = (instance_index, class_index)
            print(f'pair_key {pair_key}')
            print(self._cache.keys())
            latent, text_enc_cache = self._cache[pair_key]
            # del self._cache[pair_key]
            return latent, text_enc_cache
        else:
            print('rebuild cache')
            self._rebuilding_cache = True
            if self.pair_index > 0:
                self.highest_loss_pairs = \
                    sorted(self.loss_dict,
                           key=self.loss_dict.get,
                           reverse=True
                           )[:round(self.num_inst_images / 2)]
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
                        # self.text_encoder_cache.append(batch["input_ids"])
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


class DataLoaderType(Enum):
    CACHED_LATENTS = auto()
    REGULAR = auto()


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
            self.loss_dict = {pair: [0] for pair in self.pairs}
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

# import traceback


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        # traceback.print_stack()
        return self.latents_cache[index], self.text_encoder_cache[index]


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    logging_dir = Path(args.output_dir, "0", args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    if args.with_prior_preservation:
        pipeline = None
        for concept in args.concepts_list:
            class_images_dir = Path(concept["class_data_dir"])
            class_images_dir.mkdir(parents=True, exist_ok=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < args.num_class_images:
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                if pipeline is None:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=AutoencoderKL.from_pretrained(
                            args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
                            subfolder=None if args.pretrained_vae_name_or_path else "vae",
                            revision=None if args.pretrained_vae_name_or_path else args.revision,
                            torch_dtype=torch_dtype
                        ),
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=args.revision
                    )
                    pipeline.set_progress_bar_config(disable=True)
                    pipeline.to(accelerator.device)

                num_new_images = args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(concept["class_prompt"], num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)

                with torch.autocast("cuda"), torch.inference_mode():
                    for example in tqdm(
                        sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                    ):
                        images = pipeline(example["prompt"]).images

                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=torch.float32
    )

    def create_vae(device, weight_dtype):
        result = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
        )
        result.requires_grad_(False)
        result.to(device, dtype=weight_dtype)
        return result

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    def cache_latents(train_dataset: DreamBoothDataset,
                      train_dataloader,
                      vae
                      ) -> Tuple[DreamBoothDataset, Any]:
        latents_cache = []
        text_encoder_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                if args.train_text_encoder:
                    text_encoder_cache.append(batch["input_ids"])
                else:
                    text_encoder_cache.append(text_encoder(batch["input_ids"])[0])
        del train_dataset
        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        del train_dataloader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return train_dataset, train_dataloader

    # Move text_encode and vae to gpu (for vae it happens in create_vae).
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.

    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    vae = create_vae(accelerator.device, weight_dtype)
    # print(f'type of vae {type(vae)}')
    dreambooth_factory = DreamBoothFactory(
        collate_fn=collate_fn,
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        accelerator=accelerator,
        text_encoder=text_encoder,
        weight_dtype=weight_dtype,
        vae=vae,
        with_prior_preservation=args.with_prior_preservation,
        use_smart_cross_products=args.use_smart_cross_products,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        pad_tokens=args.pad_tokens,
        hflip=args.hflip
    )

    train_dataset = dreambooth_factory.create_dataset()
    train_dataloader = dreambooth_factory.create_dataloader(
        type=DataLoaderType.CACHED_LATENTS,
        dataset=train_dataset)

    if not args.not_cache_latents and not args.use_smart_cross_products:
        train_dataset, train_dataloader = cache_latents(train_dataset,
                                                        train_dataloader,
                                                        vae)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    number_of_examples = len(train_dataset)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {number_of_examples}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def save_weights(step):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            if args.train_text_encoder:
                text_enc_model = accelerator.unwrap_model(text_encoder)
            else:
                text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet).to(torch.float16),
                text_encoder=text_enc_model.to(torch.float16),
                vae=AutoencoderKL.from_pretrained(
                    args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
                    subfolder=None if args.pretrained_vae_name_or_path else "vae",
                    revision=None if args.pretrained_vae_name_or_path else args.revision,
                ),
                safety_checker=None,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision=args.revision,
            )
            save_dir = os.path.join(args.output_dir, f"{step}")
            pipeline.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "args.json"), "w") as f:
                json.dump(args.__dict__, f, indent=2)

            if args.save_sample_prompt is not None:
                pipeline = pipeline.to(accelerator.device)
                g_cuda = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                pipeline.set_progress_bar_config(disable=True)
                sample_dir = os.path.join(save_dir, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                with torch.inference_mode():
                    for i in tqdm(range(args.n_save_sample), desc="Generating samples"):
                        images = pipeline(
                            args.save_sample_prompt,
                            negative_prompt=args.save_sample_negative_prompt,
                            guidance_scale=args.save_guidance_scale,
                            num_inference_steps=args.save_infer_steps,
                            generator=g_cuda
                        ).images
                        images[0].save(os.path.join(sample_dir, f"{i}.png"))
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"[*] Weights saved at {save_dir}")
            unet.to(torch.float32)
            text_enc_model.to(torch.float32)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    loss_avg = AverageMeter()
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    for epoch in range(args.num_train_epochs):
        unet.train()

        if args.train_text_encoder:
            text_encoder.train()

        # if args.shuffle_after_epoch and (global_step >= number_of_examples):
        #     if vae is None:
        #         vae = create_vae(accelerator.device, weight_dtype)
        #     del train_dataset
        #     del train_dataloader
        #     train_dataset = dreambooth_factory.create_dataset()
        #     train_dataloader = dreambooth_factory.create_dataloader(train_dataset)
        #     # ,
        #     #                                      train_batch_size=args.train_batch_size,
        #     #                                      collate_fn=collate_fn
        #     #                                      )
        #     train_dataset, train_dataloader = cache_latents(train_dataset,
        #                                                     train_dataloader,
        #                                                     vae)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    if not args.not_cache_latents:
                        latent_dist = batch[0][0]
                    else:
                        latent_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
                    latents = latent_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with text_enc_context:
                    if not args.not_cache_latents:
                        if args.train_text_encoder:
                            encoder_hidden_states = text_encoder(batch[0][1])[0]
                        else:
                            encoder_hidden_states = batch[0][1]
                    else:
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = (
                #         itertools.chain(unet.parameters(), text_encoder.parameters())
                #         if args.train_text_encoder
                #         else unet.parameters()
                #     )
                #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.detach_(), bsz)

            if not global_step % args.log_interval:
                logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step > 0 and not global_step % args.save_interval and global_step >= args.save_min_steps:
                save_weights(global_step)

            progress_bar.update(1)
            global_step += 1

            if global_step >= args.max_train_steps:
                break

            train_dataset.add_loss_for_last_pair(loss.item() * bsz)

        accelerator.wait_for_everyone()

    print(train_dataset.loss_dict)
    list_sorted = sorted(train_dataset.loss_dict, key=train_dataset.loss_dict.get, reverse=True)[:5]
    print('----------------------------------------------------')
    print(list_sorted)
    save_weights(global_step)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
