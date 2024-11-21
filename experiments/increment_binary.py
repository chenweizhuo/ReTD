import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import time
from transformers import ViTForImageClassification

import os
from dataloaders import binary_dataload
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

    def forward(self, x):
        # Input size: (3, 512, 512)
        x = self.pool(F.relu(self.conv1(x)))  # (64, 256, 256)
        x = self.pool(F.relu(self.conv2(x)))  # (128, 128, 128)
        x = self.pool(F.relu(self.conv3(x)))  # (256, 64, 64)
        x = self.pool(F.relu(self.conv4(x)))  # (3, 32, 32)
        x = self.upsample(x)  # (3, 224, 224)
        return x


def min_max_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def evaluate(model, pyramid_pooling, dataloader, device):
    model.eval()
    pyramid_pooling.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for difference_images, labels in dataloader:
            difference_images = difference_images.to(device).float()
            labels = labels.to(device).float()
            pooled_images = pyramid_pooling(difference_images)

            normalized_images = min_max_normalize(pooled_images)

            outputs = model(normalized_images).logits
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    predicted_labels = torch.argmax(all_outputs, dim=1)
    accuracy = (predicted_labels == all_labels).float().mean().item()
    precision = precision_score(all_labels, predicted_labels, average='binary')
    recall = recall_score(all_labels, predicted_labels, average='binary')
    f1 = f1_score(all_labels, predicted_labels, average='binary')
    print("")
    return accuracy, precision, recall, f1


def train_transformer(train_dataloader, dev_dataloader, output_dir, model_name, epochs, int):
    print(f"生成数据为{int}个。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained("experiments/models_load/vit-base-patch16-224",
                                                      num_labels=2,
                                                      ignore_mismatched_sizes=True
                                                      ).to(device)
    pyramid_pooling = SimpleCNN().to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        pyramid_pooling = torch.nn.DataParallel(pyramid_pooling)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(pyramid_pooling.parameters()), lr=0.0001)

    model.train()
    pyramid_pooling.train()

    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        for difference_images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} using {model_name}"):
            difference_images, labels = difference_images.to(device).float(), labels.to(device).float()
            pooled_images = pyramid_pooling(difference_images)

            normalized_images = min_max_normalize(pooled_images)

            outputs = model(normalized_images)
            logits = outputs.logits
            labels_one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=2).float()
            loss = criterion(logits, labels_one_hot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} using {model_name}, Loss: {running_loss / len(train_dataloader)}")


    end_time = time.time()  # 记录训练结束时间
    training_duration = end_time - start_time  # 计算训练时长
    print(f"Training time for {model_name}: {training_duration / 60:.2f} minutes")

    model_save_path = output_dir / f"vit_model_{model_name}_only_diff_{int}.pth"
    torch.save(model.state_dict(), model_save_path)
    torch.save(pyramid_pooling.state_dict(), output_dir / f"cnn_pooling_{model_name}_only_diff_{int}.pth")
    print(f"Model and Pyramid Pooling saved to {model_save_path}")


def inference_transformer(dataloader, model_path, pyramid_pooling_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained("experiments/models_load/vit-base-patch16-224",
                                                      num_labels=2,
                                                      ignore_mismatched_sizes=True).to(
        device)
    pyramid_pooling = SimpleCNN().to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        pyramid_pooling = torch.nn.DataParallel(pyramid_pooling)

    model.load_state_dict(torch.load(model_path))
    pyramid_pooling.load_state_dict(torch.load(pyramid_pooling_path))
    model.eval()
    pyramid_pooling.eval()

    all_outputs = []
    with torch.no_grad():
        for difference_images, _ in tqdm(dataloader, desc=f"Inference using {model_name}"):
            difference_images = difference_images.to(device).float()
            pooled_images = pyramid_pooling(difference_images)

            normalized_images = min_max_normalize(pooled_images)

            outputs = model(normalized_images).logits
            all_outputs.append(outputs.cpu())

    return torch.cat(all_outputs)



def main(args):
    output_dir = Path("output/model") / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_data_base_dir = Path("preprocessed_data/only_diff")

    repo_id = "./CompVis/stable-diffusion-2-base"
    print(f"使用{repo_id}重构特征数据集训练")
    j = [1000]
    for i in j:
        model_output_dir = output_dir / repo_id.replace('/', '_').replace('.', '') / f'increment/binary/only_diff_{i}'
        model_output_dir.mkdir(parents=True, exist_ok=True)

        train_dataloader, dev_dataloader, infer_dataloader = binary_dataload(repo_id, preprocessed_data_base_dir,
                                                                             args.batch_size, i)

        # Train Transformer model
        model_name = repo_id.replace('/', '_').replace('.', '')  # 使用哪个重构编码器
        train_transformer(train_dataloader, dev_dataloader, model_output_dir, model_name, epochs=args.epochs, int=i)

        # Perform inference
        model_path = model_output_dir / f"vit_model_{model_name}_only_diff_{i}.pth"
        pyramid_pooling_path = model_output_dir / f"cnn_pooling_{model_name}_only_diff_{i}.pth"
        inference_outputs = inference_transformer(infer_dataloader, model_path, pyramid_pooling_path, model_name)

        # Assuming binary classification, calculate accuracy, precision, recall, and F1 score
        true_labels = []
        for _, labels in infer_dataloader:
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

        torch.save(inference_outputs, model_output_dir / f"inference_outputs_{model_name}_only_diff_{i}.txt")
        print(f"Inference outputs saved to {model_output_dir / f'inference_outputs_{model_name}_only_diff_{i}.txt'}")



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
