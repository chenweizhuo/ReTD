from torch.utils.data import DataLoader, random_split

from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
import random


class PreprocessedDataset(Dataset):
    def __init__(self, data_paths: List[Path]):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.load(self.data_paths[idx])


def binary_dataload(repo_id, preprocessed_data_base_dir, args_batch_size):
    if repo_id == './CompVis/SD-v1-4':
        data_id = '_CompVis_SD-v1-4'
        preprocessed_data_dir = preprocessed_data_base_dir / data_id
    else:
        data_id = '_CompVis_stable-diffusion-2-base'
        preprocessed_data_dir = preprocessed_data_base_dir / data_id

    if not preprocessed_data_dir.exists():
        raise ValueError(f"Preprocessed data directory {preprocessed_data_dir} does not exist.")
    data_amounts = {
        'adm': 50,
        'sdv5': 50,
        'biggan': 50,
        'vqdm': 50,
        'real': 500,
        'stylegan': 50,
        'wukong': 50,
        'kandinsky': 50,
        'midjourney': 50,
        'glide': 50,
        'sdv2': 50,
    }

    train_data_paths = []
    test_data_paths = []

    for subfolder, amount in data_amounts.items():
        subfolder_path = preprocessed_data_dir / subfolder
        all_files = list(subfolder_path.glob("data_*.pt"))

        if len(all_files) < amount:
            print(
                f"Not enough data in {subfolder_path}. Requested {amount}, but only found {len(all_files)}. Loading all available data.")
            amount = len(all_files)

        train_files = all_files[:50]
        remaining_files = all_files[50:]
        if subfolder == 'real':
            test_amount = 5000
        else:
            test_amount = 500
        test_files = random.sample(remaining_files, min(test_amount, len(remaining_files)))

        train_data_paths.extend(train_files)
        test_data_paths.extend(test_files)
        print(f"加载{subfolder}测试数据{test_amount}个")

        # Load datasets
    train_dataset = PreprocessedDataset(train_data_paths)
    infer_size = 4 / 5
    test_dataset = PreprocessedDataset(test_data_paths)
    dev_dataset, test_dataset = train_test_split(test_dataset, test_size=infer_size, random_state=42)

    train_dataloader = DataLoader(train_dataset, batch_size=args_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args_batch_size, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args_batch_size,
                                shuffle=False)

    return train_dataloader, dev_dataloader, test_dataloader
