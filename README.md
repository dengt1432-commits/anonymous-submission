# AdaVSkip

## Environment Setup

Linux x86_64, Python 3.12, and CUDA 12.8. Run the following commands from the project root:

```bash
conda create -n adavskip python=3.12 -y
conda activate adavskip
python -m pip install -U pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
export PYTHON_BIN="$(command -v python)"
```

## Evaluation

```bash
# Run the full MME evaluation
bash lmms-eval-main/examples/models/llava_1_5.sh
```

By default, evaluation uses `liuhaotian/llava-v1.5-7b` and `router_weight/16_1.5.safetensors`. Results are saved in `log/`.
