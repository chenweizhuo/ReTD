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
- Most results are automatically cached to avoid repeated computations (the `cache` directory can grow quite large).
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

The extracted reconstructed features are recommended to be saved in the following format
```
data
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



