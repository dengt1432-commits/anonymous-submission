# AdaVSkip

## 环境配置

Linux x86_64、Python 3.12、CUDA 12.8。以下命令在项目根目录执行：

```bash
conda create -n adavskip python=3.12 -y
conda activate adavskip
python -m pip install -U pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
export PYTHON_BIN="$(command -v python)"
```

## 测试

```bash
# 完整 MME 评测
bash lmms-eval-main/examples/models/llava_1_5.sh
```

默认使用 `liuhaotian/llava-v1.5-7b` 和 `router_weight/16_1.5.safetensors`，结果保存在 `log/`。
