
from torch.utils.data import DataLoader, random_split

from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Tuple, Dict
import random


class PreprocessedDataset_user_defined(Dataset):
    def __init__(self, data_dir: Path, data_info: Dict[str, Tuple[int, int]]):
        self.data_paths = []
        self.labels = []

        for subfolder, (amount, label) in data_info.items():
            subfolder_path = data_dir / subfolder
            all_files = list(subfolder_path.glob("data_*.pt"))
            if len(all_files) < amount:
                print(
                    f"The data in folder {subfolder_path} is insufficient. Requested {amount}, but only found {len(all_files)}. Loading all available data.")
                amount = len(all_files)
            sampled_files = random.sample(all_files, amount)
            self.data_paths.extend(sampled_files)
            self.labels.extend([label] * amount)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = torch.load(self.data_paths[idx])
        features = data[0]  # 假设特征是第一个元素
        label = self.labels[idx]
        return features, label

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir: Path, data_amounts: Dict[str, int]):
        self.data_paths = []
        for subfolder, amount in data_amounts.items():
            subfolder_path = data_dir / subfolder
            all_files = list(subfolder_path.glob("data_*.pt"))
            if len(all_files) < amount:
                print(
                    f"Not enough data in {subfolder_path}. Requested {amount}, but only found {len(all_files)}. Loading all available data.")
                amount = len(all_files)
            self.data_paths.extend(random.sample(all_files, amount))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.load(self.data_paths[idx])

def multi_dataload(repo_id, preprocessed_data_base_dir, args_batch_size, i):
    print(f"Trained by  {repo_id} reconstruction datasets")

    if repo_id == './CompVis/SD-v1-4':
        data_id = '_CompVis_SD-v1-4'
        preprocessed_data_dir = preprocessed_data_base_dir / data_id
    else:
        data_id = '_CompVis_stable-diffusion-2-base'
        preprocessed_data_dir = preprocessed_data_base_dir / data_id


        if not preprocessed_data_dir.exists():
            raise ValueError(f"Preprocessed data directory {preprocessed_data_dir} does not exist.")
    train_amounts = {
        'real_train': (i, 0),
        'adm_train': (i, 1),
        'wukong_train': (i, 2),
        'midjourney_train': (i, 3),
        'biggan_train': (i, 4),
        'vqdm_train': (i, 5),
        'glide_train': (i, 6),
        'stylegan_train': (i, 7),
        'progan_train': (i, 8),
        'dalle3_train': (i, 9),
        'sdv5_train': (i, 10),

    }



    test_amounts = {
        'real_train': (1000, 0),
        'adm': (1000, 1),
        'wukong': (1000, 2),
        'midjourney': (1000, 3),
        'biggan': (1000, 4),
        'vqdm': (1000, 5),
        'glide': (1000, 6),
        'stylegan': (1000, 7),
        'progan': (1000, 8),
        'dalle3': (1000, 9),
        'sdv5': (1000, 10),
    }

    train_dataset = PreprocessedDataset_user_defined(preprocessed_data_dir, train_amounts)
    infer_dataset = PreprocessedDataset_user_defined(preprocessed_data_dir, test_amounts)

    # Split dataset into training, development, and inference sets

    infer_size = 4/5
    dev_dataset, infer_dataset = train_test_split(infer_dataset, test_size=infer_size, random_state=42)
    train_dataloader = DataLoader(train_dataset, batch_size=args_batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args_batch_size, shuffle=False)
    infer_dataloader = DataLoader(infer_dataset, batch_size=args_batch_size, shuffle=False)
    return train_dataloader, dev_dataloader, infer_dataloader
    # return infer_dataloader


def binary_dataload(repo_id, preprocessed_data_base_dir, args_batch_size, i):
    j = i * 10
    if repo_id == './CompVis/SD-v1-4':
        data_id = '_CompVis_SD-v1-4'
        preprocessed_data_dir = preprocessed_data_base_dir / data_id
    else:
        data_id = '_CompVis_stable-diffusion-2-base'
        preprocessed_data_dir = preprocessed_data_base_dir / data_id

    if not preprocessed_data_dir.exists():
        raise ValueError(f"Preprocessed data directory {preprocessed_data_dir} does not exist.")
    train_amounts = {

        'adm_train': i,
        'sdv5_train': i,
        'biggan_train': i,
        'vqdm_train': i,
        'real_train': j,
        'stylegan_train': i,
        'wukong_train': i,
        # 'kandinsky_train': 2000,
        'midjourney_train': i,
        'glide_train': i,
        'dalle3_train': i,
        'progan_train': i

    }
    test_amounts = {
        'adm': 500,
        'sdv5': 500,
        'biggan': 500,
        'vqdm': 500,
        'real': 5000,
        'stylegan': 500,
        'wukong': 500,
        # 'kandinsky': 500,
        'midjourney': 500,
        'glide': 500,
        'dalle3': 500,
        'progan': 500
    }

    # Load preprocessed dataset
    train_dataset = PreprocessedDataset(preprocessed_data_dir, train_amounts)
    infer_dataset = PreprocessedDataset(preprocessed_data_dir, test_amounts)

    # Split dataset into training, development, and inference sets

    infer_size = 1/2
    dev_dataset, infer_dataset = train_test_split(infer_dataset, test_size=infer_size, random_state=42)
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total development samples: {len(dev_dataset)}")
    print(f"Total inference samples: {len(infer_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=args_batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args_batch_size, shuffle=False)
    infer_dataloader = DataLoader(infer_dataset, batch_size=args_batch_size, shuffle=False)
    return train_dataloader, dev_dataloader, infer_dataloader


def cross_domain_dataload(repo_id, preprocessed_data_base_dir, args_batch_size):
    if repo_id == './CompVis/SD-v1-4':
        data_id = '_CompVis_SD-v1-4'
        preprocessed_data_dir = preprocessed_data_base_dir / data_id
    else:
        data_id = '_CompVis_stable-diffusion-2-base'
        preprocessed_data_dir = preprocessed_data_base_dir / data_id

    if not preprocessed_data_dir.exists():
        raise ValueError(f"Preprocessed data directory {preprocessed_data_dir} does not exist.")
    train_amounts = {
        'real_train': 3000,
        'adm_train': 3000,

    }

    test_amounts = {
        'real': 1000,
        'sdv5': 1000,
    }

    train_dataset = PreprocessedDataset(preprocessed_data_dir, train_amounts)
    infer_dataset = PreprocessedDataset(preprocessed_data_dir, test_amounts)

    # Split dataset into training, development, and inference sets


    print(f"Total training samples: {len(train_dataset)}")

    print(f"Total inference samples: {len(infer_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=args_batch_size, shuffle=True)
    infer_dataloader = DataLoader(infer_dataset, batch_size=args_batch_size, shuffle=False)
    return train_dataloader, infer_dataloader

