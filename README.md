# DD-RUO

Dataset Distillation with Rate-Utilization Optimization. 将数据集蒸馏与神经图像压缩结合，用 TensorPool（latent + synthesis network）表达合成数据，再接入不同蒸馏方法（TM/DM/DC）。

三种蒸馏方法共享同一条 TensorPool 压缩链路，分三阶段：

1. **阶段一：合成数据集训练** — `TM/scripts/run_pool_tm.sh` / `DM/scripts/run_pool_dm.sh` / `DC/scripts/run_pool_dc.sh`
   - 入口分别为 `pool_tm.py` / `pool_dm.py` / `pool_dc.py`
   - `pool_path=init` 时，若 `pool_init.pt` 不存在会自动 `init_from_data`（warmup）再续训
   - 产出 `pool_best.pt`（TM）或 `*_exp{exp}_{it}.pt` checkpoint
2. **阶段二：后量化评估** — `quantize/scripts/run_quantize.sh` → `quantize_pool.py`
   - **TM/DM/DC 共享**，与蒸馏方法无关，只吃 pool checkpoint
   - 产出量化后的 `images_{mse}.pt` + `labels_{mse}.pt`（阶段三输入）
3. **阶段三：多架构测评** — `cross_eval/scripts/run_cross_eval.sh` → `cross_evaluate.py`
   - **TM/DM/DC 共享**，只吃 `images.pt` + `labels.pt`

> TM 还需要阶段零：生成 expert trajectory buffer — `TM/scripts/run_buffer.sh`

## 环境

```bash
conda env create -f TM/scripts/environment.yml
conda activate sre2l
```

推荐环境 `sre2l`（Python 3.10 / PyTorch 2.6.0+cu124 / CUDA 12.4）。真实 ImageNet 路径：`/data1/home/ypliu/DSproject/data`（含 train/val/meta.bin），脚本默认 `/path/to/imagenet` 占位需替换。

## 预训练权重

预训练的 pool checkpoint 和量化后的合成数据集可从 HuggingFace 下载：

<!-- TODO: 发布后替换为实际链接 -->
<!-- https://huggingface.co/<your-org>/DD-RUO -->

```
checkpoints/
├── TM/                        # 阶段一产出的 pool checkpoint（量化前）
│   ├── imagenette.pt
│   └── ...
└── quantized/                 # 阶段二产出（量化后）
    ├── imagenette/
    │   ├── pool.pt            # 量化后的 TensorPool checkpoint
    │   ├── images.pt          # 量化后的合成图像
    │   └── labels.pt          # 标签
    └── ...
```

> `images.pt` 和 `labels.pt` 可以从 `pool.pt` 重新生成：加载 pool 后调用 `TensorPool.get_data()` 即可，参考 `quantize/quantize_pool.py` 中的加载和推理流程。
>
> 如果只需要复现阶段三的跨架构测评结果，下载 `quantized/` 下的 `images.pt` + `labels.pt` 即可。

### Checkpoint 兼容性（旧权重加载）

本仓库早期的压缩主干位于顶层 `ts/` 包，重构后移至 `core/ts/`。受此影响，**重构前保存的 checkpoint**（pickle 里类路径为 `ts.tensor_data_func_v6.*`）在直接 `torch.load` 时会报 `ModuleNotFoundError: No module named 'ts'`。

为此 `core/__init__.py` 内置了一个**向后兼容 shim**：在导入 `core` 时自动把 `core.ts` 整棵子树注册成 `ts` 的别名（`sys.modules['ts...'] → core.ts...`，指向同一份代码，无冗余）。这样：

- **旧 checkpoint**（`ts.*` 路径）→ 经别名自动解析，正常加载；
- **新 checkpoint**（`core.ts.*` 路径）→ 原生加载。

无需任何手动转换。只要在使用 `torch.load` 前 `import core`（各 `pool_*.py` / `quantize_pool.py` / `cross_evaluate.py` 入口都已 `from core... import`，自动触发）。若日后遇到引用了未覆盖的 `ts.*` 子模块的旧权重，报错信息会指明缺失路径，届时在 `core/__init__.py` 的别名注册处补上即可。

## 快速开始

下面以 TM 方法为例。DM/DC 只需替换阶段一的入口脚本，阶段二/三完全相同。

### 阶段零（仅 TM 需要）：生成 Expert Trajectories

阶段一（TM 训练）需要 expert trajectory buffer。如果没有现成的 buffer，先运行：

```bash
cd TM
bash scripts/run_buffer.sh
```

在 `scripts/run_buffer.sh` 顶部修改 `DATA_PATH`（ImageNet 路径）和 `SUBSET`。每个子集会生成 10 个 `replay_buffer_*.pt`，存放在 `BUFFER_PATH/ImageNet/{subset}/128/ConvNetD5/` 下。

### 阶段一：合成数据集训练

```bash
cd TM
bash scripts/run_pool_tm.sh
```

在 `scripts/run_pool_tm.sh` 顶部修改参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `cuda_id` | GPU 编号 | `0,1` |
| `dst` / `subset` | 数据集 / ImageNet 子集 | `ImageNet` / `imagenette` |
| `ipc` | 每类图像数 | `102` |
| `data_path` | 数据集路径 | `/path/to/imagenet` |
| `buffer_path` | Expert trajectory 路径 | `/path/to/buffers` |
| `pool_init` | Pool 初始化（`init`=从头训练，或 `.pt` 路径续训），传给 `--pool_path` | `init` |
| `arm` / `dim` / `layers_v` | 压缩网络结构参数 | `32` / `4` / `v5` |
| `save_path` | 结果保存目录 | `./results/tm` |

产出：`save_path/<FLAG>/pool_best.pt`（checkpoint）、`log.txt`、合成图像预览 `.png`

### 阶段二：后量化评估

```bash
cd quantize
bash scripts/run_quantize.sh
```

在 `scripts/run_quantize.sh` 顶部修改参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `CUDA_ID` | GPU 编号 | `0,1` |
| `DATASET` / `SUBSET` / `IPC` | 数据集配置（需与阶段一一致） | `ImageNet` / `imagenette` / `102` |
| `POOL_PATH` | **阶段一产出的 checkpoint 路径** | `./results/tm/.../pool_best.pt` |
| `MSE_ERR` | 量化 MSE 阈值（越小越精确，码率越高） | `0.0000005` |
| `LAYERS_V` / `ARM` / `DIM` | 压缩网络结构（需与阶段一一致） | `v5` / `32` / `4` |

产出（`SAVE_DIR/<FLAG>/` 目录下）：

| 文件 | 说明 |
|------|------|
| `pool.pt` | 量化后的 TensorPool checkpoint（可用于重新生成 images/labels） |
| `images_{mse}.pt` | 量化后的合成图像（已归一化，阶段三输入） |
| `labels_{mse}.pt` | 对应标签（阶段三输入） |
| `Synthetic_Images_*.png` | 可视化预览 |

> 注：`quantize_pool.py` 与蒸馏方法无关，TM/DM/DC 训练出的 pool checkpoint 都可以用。

### 阶段三：多架构测评

```bash
cd cross_eval
bash scripts/run_cross_eval.sh
```

在 `scripts/run_cross_eval.sh` 顶部修改参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `CUDA_ID` | GPU 编号 | `0` |
| `DATASET` / `SUBSET` | 数据集配置（需与前面阶段一致） | `ImageNet` / `imagenette` |
| `IMAGES_PATH` | **阶段二产出的合成图像 `.pt` 文件** | `./results/quantize/.../images_5e-07.pt` |
| `LABELS_PATH` | **阶段二产出的标签 `.pt` 文件** | `./results/quantize/.../labels_5e-07.pt` |
| `MODELS` | 逗号分隔的评估架构列表 | `ResNet18ImageNet,VGG11,AlexNet,ViT` |
| `NUM_EVAL` | 每个架构的重复评估次数 | `5` |
| `EPOCH_EVAL` | 评估训练的 epoch 数 | `1000` |

产出：`SAVE_DIR/log.txt` 中各架构的准确率均值和标准差

> 注：`cross_evaluate.py` 与蒸馏方法无关，任意方法产出的 `images.pt` + `labels.pt` 都可以用。

## 目录结构

```
DD-RUO/
├── core/                       # 共享库（一份）
│   ├── ts/                     # 图像压缩主干（TensorPool 核心）
│   │   ├── tensor_pool.py      # TensorPool（训练 + 量化）
│   │   ├── tensor_data_func_v6.py
│   │   ├── training.py
│   │   └── core/               # ARM / Synthesis / Quantizer 等
│   ├── utils.py                # 数据集/网络/评估工具函数（DC 超集）
│   └── networks.py             # 评估网络（ConvNet / ResNet / VGG / AlexNet / ViT）
├── TM/                         # Trajectory Matching
│   ├── buffer.py               # Expert trajectory 生成（阶段零）
│   ├── pool_tm.py              # 阶段一：训练入口
│   ├── hyper_params.py
│   ├── reparam_module.py
│   └── scripts/
├── DM/                         # Distribution Matching
│   ├── pool_dm.py
│   └── scripts/
├── DC/                         # Gradient Matching
│   ├── pool_dc.py
│   └── scripts/
├── quantize/                   # 阶段二：后量化评估（TM/DM/DC 共享）
│   ├── quantize_pool.py
│   └── scripts/
├── cross_eval/                 # 阶段三：多架构测评（TM/DM/DC 共享）
│   ├── cross_evaluate.py
│   └── scripts/
└── checkpoints/                # 预训练权重（HuggingFace 下载）
    ├── TM/
    └── quantized/
```

## 代码关系

1. `core/` 是共享库：`ts/`（图像压缩主干）、`utils.py`、`networks.py`，三种方法共用一份。
2. `TM/`、`DM/`、`DC/` 三个入口把各自的蒸馏方法（轨迹 / 分布 / 梯度匹配）与 TensorPool 压缩表达结合。
3. `quantize/` 与 `cross_eval/` 与蒸馏方法无关，任意方法产出的 pool checkpoint / images+labels 都能用。
