# SimVLA: A Simple VLA Baseline for Robotic Manipulation

| **Paper** | **Website** | **Model & Data** |
| :------------------: | :-----------------------: | :---------------------: |
| [![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18224) | [![Website](https://img.shields.io/badge/Project%20Page-181717?style=for-the-badge&logo=githubpages&logoColor=white)](https://frontierrobo.github.io/SimVLA/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFBA00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/YuankaiLuo/simvla) |

A simple and efficient Vision-Language-Action (VLA) model for robot manipulation tasks.

<img width="506" height="796" alt="image" src="https://github.com/user-attachments/assets/7ffb8969-aa4f-4bcc-8c38-33d5e7da4b25" />

### 1. Create Conda Environment

```bash
conda create -n simvla python=3.10 -y
conda activate simvla
```

### 2. Install PyTorch (CUDA 12.4)

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Core Dependencies

```bash
pip install transformers==4.57.3
pip install peft==0.19.1 accelerate==1.13.0 tensorboard==2.20.0 safetensors==0.7.0 scipy==1.15.3 einops==0.8.2 timm==1.0.27
```

### 4. Install Data & IO Libraries

```bash
pip install mmengine==0.10.7 pyarrow==24.0.0 h5py==3.14.0 av==17.0.1 opencv-python==4.10.0.84 imageio==2.37.3 Pillow==12.2.0
```

### 5. Install Training & Serving Tools

```bash
pip install fastapi==0.136.1 uvicorn==0.47.0 wandb==0.27.0 json-numpy==2.1.1 msgpack-numpy==0.4.8 mediapy==1.2.6 num2words==0.5.14 websockets==16.0
```

> **Note**: `flash-attn` is **not required** — it is not imported anywhere in the project and can safely be skipped.

### Verified Environment

| Package | Version |
|---------|---------|
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| transformers | 4.57.3 |
| accelerate | 1.13.0 |
| peft | 0.19.1 |

## Training (LIBERO Dataset)

### 1. Prepare LIBERO Dataset

Download [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) dataset, and place it in `./datasets/metas/`.

### 2. Create Training Metadata

```bash
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json
```

### 3. Compute Normalization Statistics

```bash
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```
### 4.下载 smolVLM-500M-Instruct 到 pretrained 目录
```
modelscope download --model HuggingFaceTB/SmolVLM-500M-Instruct --local_dir pretrained
```

### 5. Start Training

**Small Model Configuration:**
```bash
bash train_smolvlm_small.sh
```

**Large Model Configuration:**
```bash
bash train_smolvlm_large.sh
```

### 6. Evaluation

```bash
cd evaluation/libero
```

### 7. Results

<img width="506" height="1220" alt="image" src="https://github.com/user-attachments/assets/6ee1cd5e-42c5-4cf7-9cce-6dc04c1a215f" />

## Model Architecture

- **Vision-Language Backbone**: SmolVLM-500M-Instruct (576 hidden dim)
- **Action Transformer**: Configurable depth and width
  - Small: 768 hidden, 12 layers, 12 heads
  - Large: 1024 hidden, 24 layers, 16 heads

## Reference

If you find our codes useful, please consider citing our work

```
@article{luo2026simvla,
  title={SimVLA: A Simple VLA Baseline for Robotic Manipulation},
  author={Luo, Yuankai and Chen, Woping and Liang, Tong and Wang, Baiqiao and Li, Zhenguo},
  journal={arXiv preprint arXiv:2602.18224},
  year={2026}
}
```


