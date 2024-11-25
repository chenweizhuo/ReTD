# ReTD: Reconstruction-Based Traceability Detection for Generated Images


## Setup
Create a virtual environment and run
```
pip install -r requirements.txt
```
(tested with Python 3.10).


## Reproducing our Experiments
Here we provide the commands to reproduce our experimental results.
Note the following:
- Your results might differ slightly due to randomness and different library versions.
- Code to reconstruct the tables and figures is provided in the notebooks corresponding to the script names.
- Since we are using an offline environment for our experiments, we need to load the pre-trained model locally for use. Or you can change the code to access hugging face directly.
- All reconstruction featrues are saved in `preprocessed_data`, in unique directories for each configuration.

### Data

Our generated images data can be downloaded from [GenImage](https://github.com/GenImage-Dataset/GenImage). Extract the `.zip` file and place the `data` directory inside the root directory of this repository. The real folder contains the collected images from ImageNet.
```
data
├── real
├── adm
├── biggan
├── midjourney
└── ...
```

### Reconstruction
Here the reconstruction module and the discrimination module in the model are split in order to facilitate the understanding of the role of the two modules. All reconstruction featrues are saved in `preprocessed_data`, in unique directories for each configuration.

The version of VAE is `stable-diffusion-2-base`, which can be downloaded locally from huggingface and saved in path `CompVis/stable-diffusion-2-base`.
To extract reconstruction feature, you can run
```
python data_process_diff.py
```
Reconstruction with no automation set up, the file paths in the code need to be changed to enable processing of all categories.

The extracted reconstructed features are recommended to be saved in the following format
```
preprocessed_data
└── only_diff
    └── _CompVis_stable-diffusion-2-base
        ├── real
        ├── real_train
        ├── adm
        ├── adm_train
        ├── biggan
        ├── biggan_train
        ├── midjourney
        ├── midjourney_train
        └── ...

```

### Train Discriminative Module

ViT can be downloaded locally from the hugging face website and saved in path `experiments/models_load/vit-base-patch16-224`.

You can change the code in `dataloaders.py` to control the amount of training data and test data.

For the traceability experiment, you can train Discriminative Module by running 
```
experiments/multi_class.py
```
For the binary detection experiment, you can train Discriminative Module by running 
```
experiments/binary_class.py
```
For the Incremental experiment, you can train Discriminative Module by running 
```
experiments/increment_class.py
```
If you want to do data imbalance detection, you can change the number of test data for the `multi_dataload` function in `dataloader.py` and run
```
experiments/multi_test.py
```
It is worth noting that when executing the test file code, be sure to determine the path to the trained parameters.
