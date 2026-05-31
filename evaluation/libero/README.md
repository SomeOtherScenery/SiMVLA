# Evaluation on LIBERO

## 1. Environment Setup

Set up LIBERO following the [official instructions](https://github.com/Lifelong-Robot-Learning/LIBERO).

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

## 2. Start Server

```
conda activate simvla
CUDA_VISIBLE_DEVICES=1 python ./evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/simvla_libero_small/ckpt-180000 \
    --norm_stats ./norm_stats/libero_norm.json \
    --smolvlm_model ./pretrained/SmolVLM-500M-Instruct \
    --port 8102
```

## 3. Run Evaluation

Quick evaluation on selected tasks:

Full evaluation on all task suites:

```bash
conda activate libero
bash run_eval_all.sh
```
