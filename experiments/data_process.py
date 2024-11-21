import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as tf
from torchvision.transforms import transforms
from diffusers import StableDiffusionPipeline
from diffusers.models import VQModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from transformers import CLIPTokenizer, ViTImageProcessor

from retd.misc import device
from typing import List, Tuple
from PIL import Image

from torchvision.transforms.functional import to_pil_image  # Import to_pil_image
from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2
import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def compute_local_entropy(image: np.ndarray, disk_size: int = 5) -> np.ndarray:
    """Compute the local entropy of a grayscale image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local_entropy = entropy(gray_image, disk(disk_size))
    return local_entropy


def high_pass_filter(image):
    # Split the image into separate channels
    b, g, r = cv2.split(image)

    for i in range(1, 11):
        kernel = np.array([
            [-1, -1, -1, -1, -1],
            [-1, 1, 2, 1, -1],
            [-1, 2, 4, 2, -1],
            [-1, 1, 2, 1, -1],
            [-1, -1, -1, -1, -1]
        ])
        b_filtered_image = cv2.filter2D(b, -1, kernel)
        g_filtered_image = cv2.filter2D(g, -1, kernel)
        r_filtered_image = cv2.filter2D(r, -1, kernel)

    # Merge the filtered channels back into an image
    filtered_image = cv2.merge((b_filtered_image, g_filtered_image, r_filtered_image))
    return filtered_image


def image_reconstruction(
        original_path: Path,
        repo_id: str,
        output_dir: Path,
        seed: int = 1,
) -> Path:
    """Compute AE reconstruction for an image and save it."""

    # Set up pipeline
    clip_model_path = "CompVis/clip-vit-base-patch16"
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
    pipe = StableDiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.float32,  # Use float32 for CPU
        tokenizer=tokenizer,
    )
    pipe.to("cpu")  # Move the pipeline to CPU

    # Extract AE
    ae = pipe.vae if hasattr(pipe, "vae") else pipe.movq
    ae.to("cpu")  # Move the autoencoder to CPU
    ae = torch.compile(ae)
    decode_dtype = next(iter(ae.post_quant_conv.parameters())).dtype

    # Reconstruct
    generator = torch.Generator().manual_seed(seed)

    # Load original image
    original_image = Image.open(original_path).convert("RGB")
    # Normalize
    image_tensor = torch.unsqueeze(
        torch.tensor(transforms.ToTensor()(original_image)).to("cpu", dtype=ae.dtype),
        dim=0
    ) * 2.0 - 1.0

    # Encode
    latent = retrieve_latents(ae.encode(image_tensor), generator=generator)

    # Decode
    if isinstance(ae, VQModel):
        reconstruction = ae.decode(
            latent.to(decode_dtype), force_not_quantize=True, return_dict=False
        )[0]
    else:
        reconstruction = ae.decode(
            latent.to(decode_dtype), return_dict=False
        )[0]

    # De-normalize
    reconstruction = (reconstruction / 2 + 0.5).clamp(0, 1)

    # Save
    reconstruction_path = output_dir / f"{original_path.stem}.png"
    to_pil_image(reconstruction.squeeze(0)).save(reconstruction_path)
    del pipe, ae, latent, reconstruction, image_tensor, original_image
    torch.cuda.empty_cache()

    return reconstruction_path

class SingleImageDataset(Dataset):
    def __init__(self, original_paths: List[Path], feature_extractor, labels: List[int], repo_id: str,
                 seed: int, num_workers: int, output_dir: Path, preprocessed_data_dir: Path):
        self.original_paths = original_paths
        self.feature_extractor = feature_extractor
        self.labels = labels
        self.repo_id = repo_id
        self.seed = seed
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.preprocessed_data_dir = preprocessed_data_dir

    def preprocess_and_save(self):
        for idx in range(len(self.original_paths)):
            original_path = self.original_paths[idx]
            print(f"读取图片路径为：{original_path}")
            label = self.labels[idx]

            original_image = Image.open(original_path).convert("RGB")
            original_image = original_image.resize((512, 512), Image.LANCZOS)
            reconstruction_image_path = image_reconstruction(original_path, self.repo_id, self.output_dir, self.seed)
            reconstruction_image = Image.open(reconstruction_image_path).convert("RGB")
            reconstruction_image = reconstruction_image.resize((512, 512), Image.LANCZOS)

            original_np = np.array(original_image)
            local_entropy = compute_local_entropy(original_np)
            high_pass = high_pass_filter(original_np)

            local_entropy_tensor = tf.ToTensor()(local_entropy)
            high_pass_tensor = tf.ToTensor()(high_pass)
            original_tensor = tf.ToTensor()(original_image)
            reconstruction_tensor = tf.ToTensor()(reconstruction_image)
            local_entropy_tensor = local_entropy_tensor.repeat_interleave(3, dim=0)
            '''
            print(f"reconstruction_tensor size:{reconstruction_tensor.size()}")
            print(f"original_tensor size:{local_entropy_tensor.size()}")
            print(f"local_entropy_tensor size:{local_entropy_tensor.size()}")
            print(f"high_pass_tensor size:{high_pass_tensor.size()}")

            '''

            difference_tensor = original_tensor - reconstruction_tensor

            # target_size = (224, 224)
            # difference_tensor = TF.resize(difference_tensor, target_size)
            # local_entropy_tensor = TF.resize(local_entropy_tensor, target_size)
            # high_pass_tensor = TF.resize(high_pass_tensor, target_size)

            combined_features = torch.tensor(difference_tensor + local_entropy_tensor + high_pass_tensor)
            # combined_features = torch.clamp(combined_features, 0, 1)


            torch.save((combined_features, torch.tensor(label, dtype=torch.float32)),
                       self.preprocessed_data_dir / f"data_{idx}.pt")

    def __len__(self):
        return len(self.original_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.load(self.preprocessed_data_dir / f"data_{idx}.pt")


class PreprocessedDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_paths = list(data_dir.glob("data_*.pt"))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.load(self.data_paths[idx])


def main(args):
    dirs = [args.real_dir] + args.fake_dirs
    labels = [
        0 if Path(dir).resolve() == Path(args.real_dir).resolve() else 1
        for dir in dirs for file in Path(dir).rglob('*') if file.is_file()
    ]
    original_paths = [
        file for dir in dirs for file in Path(dir).rglob('*') if file.is_file()
    ]

    for repo_id in args.repo_ids:
        output_dir = Path("output/res") / repo_id
        output_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_data_dir = Path("preprocessed_data/real") / "real_type"
        preprocessed_data_dir.mkdir(parents=True, exist_ok=True)

        feature_extractor = ViTImageProcessor.from_pretrained("experiments/models_load/vit-base-patch16-224")
        paired_dataset = SingleImageDataset(
            original_paths,
            feature_extractor,
            labels,
            repo_id,
            args.seed,
            args.num_workers,
            output_dir,
            preprocessed_data_dir
        )

        # Preprocess and save data
        paired_dataset.preprocess_and_save()

        print(f"Data preprocessing complete and saved for {repo_id}.")
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="default")

    # images
    parser.add_argument("--precomputed-real-dist", type=Path)
    parser.add_argument("--real-dir", type=Path, default="data/raw/real")
    parser.add_argument(
        "--fake-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("data/raw/generated/CompVis-stable-diffusion-v1-1-ViT-L-14-openai"),
            Path("data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai"),
            Path(
                "data/raw/generated/stabilityai-stable-diffusion-2-1-base-ViT-H-14-laion2b_s32b_b79k"
            ),
            Path(
                "data/raw/generated/kandinsky-community-kandinsky-2-1-ViT-L-14-openai"
            ),
            Path("data/raw/generated/midjourney-v4"),
            Path("data/raw/generated/midjourney-v5"),
            Path("data/raw/generated/midjourney-v5-1"),
            Path("data/raw/generated/imagenet_ai_0419_biggan"),
            Path("data/raw/generated/imagenet_ai_0419_vqdm"),
            Path("data/raw/generated/imagenet_ai_0424_sdv5"),
            Path("data/raw/generated/imagenet_ai_0424_wukong"),
            Path("data/raw/generated/imagenet_ai_0508_adm")
        ],
    )
    parser.add_argument("--amount", type=int)
    parser.add_argument("--transforms", nargs="*", default=["clean"])
    parser.add_argument(
        "--reconstruction-root", type=Path, default="data/reconstructions"
    )

    # autoencoder
    parser.add_argument(
        "--repo-ids",
        nargs="+",
        default=[
            "./CompVis/SD-v1-4",
            "./CompVis/stable-diffusion-2-base",
            # "./CompVis/kandinsky-2-1",
        ],
    )

    # distance
    parser.add_argument(
        "--distance-metrics",
        nargs="+",
        default=[
            "lpips_vgg_-1",
        ],
    )

    # technical
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train_test the model")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
