# Dataset Distillation as Data Compression (DD-RUO) [ICCV 2025]

本仓库是论文 **Dataset Distillation as Data Compression: A Rate-Utility Perspective**（ICCV 2025）的官方 PyTorch 实现。

| [论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Bao_Dataset_Distillation_as_Data_Compression_A_Rate-Utility_Perspective_ICCV_2025_paper.pdf) | [项目主页](https://github.com/nouise/DD-RUO) | [预训练](#预训练权重) |

## Overview
<!-- ![Teaser image](overview.png) -->
> 我们将数据集蒸馏重新建模为**率-效用（rate-utility）优化**问题：合成数据集不再以像素形式存储，而是用神经图像压缩的方式参数化——每个（组）合成样本由一组隐变量（latent grids）、一个熵网络与一个解码器表示（沿用 C3 的设计）。以每像素比特数（bpp）度量存储代价，以下游训练精度度量效用，二者的权衡由系数 λ 控制。该参数化与具体蒸馏目标正交，可统一搭配 Trajectory Matching (TM)、Gradient Matching (GM)、Distribution Matching (DM) 三种损失。

## Requirements
```
conda env create -f TM/scripts/environment.yml
conda activate sre2l
```

## Dataset
下载 [ImageNet-1K](https://www.image-net.org/)，并在各脚本顶部填入数据集路径。实验在 128×128 的 ImageNet 10 类子集上进行：ImageNette / ImageWoof / ImageFruit / ImageYellow / ImageMeow / ImageSquawk。

## Method
合成数据集的训练分为四个阶段（与论文一致）：

1. **Initialization**（warm-up）：用率-失真系数 **β** 初始化压缩表示（TM 取 β=10，GM/DM 取 β=10⁶）。
2. **Joint rate-utility optimization**：用率-效用系数 **λ** 联合优化蒸馏目标与码率，采用**两阶段调度**——前半程较大的 λ 偏重蒸馏性能，后半程减小 λ 以收紧比特预算。
3. **Post-quantization**：在一组 MSE 阈值中选择能满足目标 bpc 预算且率-效用最优的量化精度。
4. **Evaluation**：训练若干个分类器并报告平均精度；跨架构评估作为消融实验。

主要超参数：

- `β` — 初始化阶段的率-失真权衡
- `λ` — 联合优化阶段的率-效用权衡（两阶段）
- 解码器版本 / 熵网络上下文大小 / slice size — 压缩网络结构

各数据集、各蒸馏损失下的详细取值见论文，逐实验的真实启动脚本、日志位置与权重映射见 [`checkpoints_release/`](checkpoints_release/README.md)。

> **关于为何有这么多 β / λ 取值**：论文中每个配置的 β / λ 都是为了**严格满足对应的码率预算（bpc）**而单独调过的。实际使用时通常无需如此——**按脚本中的推荐默认值即可获得很好的效果**；只有当你需要卡某个特定的 bpc 预算时才需要调 λ（λ 越大越偏重效用、码率越松，λ 越小码率越紧）。

## Usage
三种蒸馏损失共享同一条压缩链路，区别仅在蒸馏训练入口；脚本中 `pool_path=init` 时会自动完成 Initialization（warm-up）。

### TM（默认）
TM 需要先生成 expert trajectories：
```
cd TM/scripts
bash run_buffer.sh
bash run_pool_tm.sh
```

### DM
```
cd DM/scripts
bash run_pool_dm.sh
```

### GM（代码目录为 `DC/`）
```
cd DC/scripts
bash run_pool_dc.sh
```

### Post-quantization（三方法共享）
```
cd quantize/scripts
bash run_quantize.sh
```

### Cross-architecture Evaluation（消融，三方法共享）
```
cd cross_eval/scripts
bash run_cross_eval.sh
```

> 各阶段的超参数（β、λ、解码器结构、MSE 阈值、评估架构等）均在对应 bash 脚本顶部，请参照脚本注释填写。

## 预训练权重
预训练的合成数据集、训练日志与启动脚本发布在 [ModelScope: yiping03/dd-ruo0](https://www.modelscope.cn/models/yiping03/dd-ruo0)。下载后放入 `checkpoints/` 即可直接用于 Post-quantization 与 Evaluation：

```python
from modelscope import snapshot_download
snapshot_download('yiping03/dd-ruo0')
```

逐实验的真实启动脚本、日志位置与权重映射见 ModelScope 仓库内的 [`checkpoints_release/`](https://www.modelscope.cn/models/yiping03/dd-ruo0)。

## Citation
如果本工作对你的研究有帮助，欢迎引用：

```bibtex
@inproceedings{bao2025ruo,
  author    = {Youneng Bao and Yiping Liu and Zhuo Chen and Yongsheng Liang and Mu Li and Kede Ma},
  title     = {Dataset Distillation as Data Compression: A Rate-Utility Perspective},
  booktitle = {IEEE/CVF International Conference on Computer Vision},
  year      = {2025},
}
```

## Acknowledgement
本仓库的实现建立在以下工作之上：
- *C3: High-performance and low-complexity neural compression (Cool-Chic)* — 压缩主干（latent + 熵网络 + 解码器） — [Code](https://github.com/Orange-OpenSource/Cool-Chic)
- *Frequency Domain-based Dataset Distillation (FreD)*, NeurIPS 2023 — [Code](https://github.com/sdh0818/FreD)
- *Distilling Dataset into Neural Field (DDiF)*, ICLR 2025 — [Code](https://github.com/aailab-kaist/DDiF)
- *Dataset Condensation with Gradient Matching / Distribution Matching* — [Code](https://github.com/VICO-UoE/DatasetCondensation)
- *Dataset Distillation by Matching Training Trajectories* — [Code](https://github.com/georgecazenavette/mtt-distillation)
