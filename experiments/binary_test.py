import argparse
from pathlib import Path
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from binary_class import inference_transformer
import os
from dataloaders import binary_dataload
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main(args):
    repo_id = "./CompVis/stable-diffusion-2-base"
    print(f"使用{repo_id}重构特征数据集测试")
    output_dir = Path("output/model") / args.experiment_id
    preprocessed_data_base_dir = Path("preprocessed_data/only_diff")
    model_output_dir = output_dir / repo_id.replace('/', '_').replace('.', '') / 'binary/only_diff'  # 获得模型路径
    model_name = repo_id.replace('/', '_').replace('.', '')
    _, _, test_data_load = binary_dataload(repo_id, preprocessed_data_base_dir, args.batch_size, i=2000)
    model_path = model_output_dir / f"vit_model_{model_name}_only_diff.pth"
    pyramid_pooling_path = model_output_dir / f"cnn_pooling_{model_name}_only_diff.pth"

    inference_outputs = inference_transformer(test_data_load, model_path, pyramid_pooling_path, model_name)

    true_labels = []
    for _, labels in test_data_load:
        true_labels.extend(labels.numpy())
    true_labels = torch.tensor(true_labels)  # True labels for the inference dataset
    predicted_labels = torch.argmax(inference_outputs, dim=1)
    accuracy = (predicted_labels == true_labels).float().mean().item()
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    f1 = f1_score(true_labels, predicted_labels, average='binary')

    print(f"Accuracy for {repo_id}: {accuracy * 100:.2f}%")
    print(f"Precision for {repo_id}: {precision * 100:.2f}%")
    print(f"Recall for {repo_id}: {recall * 100:.2f}%")
    print(f"F1 Score for {repo_id}: {f1 * 100:.2f}%")

    torch.save(inference_outputs, model_output_dir / f"inference_outputs_{model_name}.txt")
    print(f"Inference outputs saved to {model_output_dir / f'inference_outputs_{model_name}.txt'}")



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
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train_test the model")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())