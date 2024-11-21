import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from retd.high_level_funcs import compute_distances
from retd.misc import safe_mkdir, write_config
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F


class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_mlp(X, y, output_dir, combination_name):
    print(f"Training model for combination: {combination_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreeLayerMLP(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Split the data into training and inference sets
    X_train, X_infer, y_train, y_infer = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_infer_tensor = torch.tensor(X_infer, dtype=torch.float32).to(device)
    y_infer_tensor = torch.tensor(y_infer, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    infer_dataset = TensorDataset(X_infer_tensor, y_infer_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    infer_dataloader = DataLoader(infer_dataset, batch_size=32, shuffle=False)

    num_epochs = 10
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())
        train_accuracy = accuracy_score(y_true_train, y_pred_train)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}, Train Accuracy: {train_accuracy * 100:.2f}%")

    # Inference phase
    model.eval()
    y_true_infer = []
    y_pred_infer = []
    with torch.no_grad():
        for inputs, labels in infer_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true_infer.extend(labels.cpu().numpy())
            y_pred_infer.extend(predicted.cpu().numpy())
    infer_accuracy = accuracy_score(y_true_infer, y_pred_infer)
    print(f"Inference Accuracy for combination {combination_name}: {infer_accuracy * 100:.2f}%")

    torch.save(model.state_dict(), output_dir / f"mlp_model_{combination_name}.pth")


def main(args):
    output_dir = Path("output/01") / args.experiment_id
    safe_mkdir(output_dir)
    write_config(vars(args), output_dir)

    # Compute distances for each combination
    if args.precomputed_real_dist is not None:
        dirs = args.fake_dirs
    else:
        dirs = [args.real_dir] + args.fake_dirs

    all_distances = compute_distances(
        dirs=dirs,
        transforms=args.transforms,
        repo_ids=args.repo_ids,
        distance_metrics=args.distance_metrics,
        amount=args.amount,
        reconstruction_root=args.reconstruction_root,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.precomputed_real_dist is not None:
        all_distances = pd.concat([all_distances, pd.read_pickle(args.precomputed_real_dist)])

    categoricals = ["dir", "image_size", "repo_id", "transform", "distance_metric", "file"]
    all_distances[categoricals] = all_distances[categoricals].astype("category")
    all_distances.to_parquet(output_dir / "all_distances.parquet")

    # Combine all distances for training and inference
    for (transform, repo_id, dist_metric), group_df in all_distances.groupby(
            ["transform", "repo_id", "distance_metric"], sort=False, observed=True
    ):
        combined_distances = []
        labels = []
        y_score_real = group_df.query("dir == @args.real_dir.__str__()").distance.values
        for fake_dir in args.fake_dirs:
            y_score_fake = group_df.query("dir == @fake_dir.__str__()").distance.values
            combined_distances.extend(y_score_real.tolist() + y_score_fake.tolist())
            labels.extend([0] * len(y_score_real) + [1] * len(y_score_fake))

        X = np.array(combined_distances).reshape(-1, 1)
        y = np.array(labels)

        # Train and evaluate a single MLP on this combination's distances
        combination_name = f"{transform}_{repo_id}_{dist_metric}".replace("/", "_")
        safe_mkdir(output_dir / combination_name)
        train_mlp(X, y, output_dir / combination_name, combination_name)


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
    parser.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
