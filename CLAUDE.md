## 定位
你是一名专业的开源代码专家，你最擅长整理开源项目和代码。现在时间紧迫，只有3个小时的时间，我们要完成代码的整理工作，好消息是我对代码非常熟悉，但是没有笔记和note,全部信息都在我的脑袋里面。
我需要你辅助或者帮助我完成这个代码整理的工作，并进行开源到github。

## 代码说明
`DD-RUO` 是一个把数据集蒸馏与图像压缩结合起来的研究代码仓库。核心思路是用 TensorPool 风格的 `latent + synthesis network` 表达合成数据，再把码率建模、量化和编解码流程接进不同的数据集蒸馏方法里。
我的代码是参考/data1/home/ypliu/reference/DDiF-main，/data1/home/ypliu/reference/FreD-main这两个项目实现的，目标也是整理成类似的样子，不同的蒸馏方法搭配同一个tensorpool对应的图像合成的思路。
下面的目录结构仅供参考。其中全部以TM的为准，其他的都是很小的改动。

我现在第一步就是要改进
1.run_pool_tm.sh，确保这个流程通常，这样就能成功合成数据集。
2.第二步就是后量化，/data1/home/ypliu/DD-RUO/TM/scripts/run_quantize_net_2.sh，注意这个似乎是对所有的TM,DM,DC都是使用的这个量化脚本，例如这些/data1/home/ypliu/DSproject/quantize_net/ImageNet/imagemeow/102/TEST_ImageNet_imagemeow_ipc102_32_4_syn40_batch_syn80_save_img/run_quantize_net_2.sh
3.就是多个架构的测评，我还没有找到代码，/data1/home/ypliu/DSproject/FreD/ImageNet-abcde/run.sh，就是这里的代码。
其实就这三个环节需要展示。然后我需要给出对应的启动脚本和已经保存好的权重。你给我把位置留出来，我去寻找。
## 三阶段 Pipeline（已整理完成）

三个蒸馏方法（TM/DM/DC）共享同一条 TensorPool 压缩链路，共享代码已去重到 `core/`，quantize/cross_eval 已拎成顶级目录。分三阶段：

1. **阶段零（仅 TM 需要）**：生成 expert trajectory buffer — `TM/scripts/run_buffer.sh`
2. **阶段一：合成数据集训练** — `TM/scripts/run_pool_tm.sh` / `DM/scripts/run_pool_dm.sh` / `DC/scripts/run_pool_dc.sh`
   - 入口分别为 `pool_tm.py` / `pool_dm.py` / `pool_dc.py`
   - `pool_path=init` 时，若 `pool_init.pt` 不存在会自动 `init_from_data`（warmup）再续训
   - 产出 `pool_best.pt`（TM）或 `*_exp{exp}_{it}.pt` checkpoint
3. **阶段二：后量化评估** — `quantize/scripts/run_quantize.sh` → `quantize/quantize_pool.py`
   - **TM/DM/DC 共享**，与蒸馏方法无关，只吃 pool checkpoint
   - 单次流程：load pool → `quantize_net(mse_err)` → evaluate → 保存 `images_{mse}.pt` + `labels_{mse}.pt` + `pool.pt`
4. **阶段三：多架构测评** — `cross_eval/scripts/run_cross_eval.sh` → `cross_eval/cross_evaluate.py`
   - **TM/DM/DC 共享**，只吃 `images_{mse}.pt` + `labels_{mse}.pt`

详细参数说明见 `README.md`。

## 运行环境

- **推荐环境：`sre2l`**（torch 2.6.0+cu124, torchvision 0.21, 8× RTX 4090 可用）
  - `README.md` 里写的 `c3` 环境 **当前 torch 已损坏**（`ImportError: undefined symbol: iJIT_NotifyEvent`，MKL 符号冲突），暂不可用，待修复或更新 README
- 真实 ImageNet 路径：`/data1/home/ypliu/DSproject/data`（含 train/val/meta.bin），脚本默认 `/path/to/imagenet` 占位需替换

## 目录结构（整理后现状）

| 路径 | 作用 |
| --- | --- |
| `core/` | 共享库（一份）：`ts/`（图像压缩主干）+ `utils.py`（DC 超集）+ `networks.py`。三方法及 quantize/cross_eval 全部用 `from core.` 导入 |
| `core/ts/` | 图像压缩主干（TensorPool 核心）：`tensor_pool.py` / `tensor_data_func_v6.py` / `training.py` / `core/`（ARM/Synthesis/Quantizer）。内部相对导入 |
| `TM/` | Trajectory Matching：`buffer.py` / `pool_tm.py` / `hyper_params.py` / `reparam_module.py` / `scripts/` |
| `DM/` | Distribution Matching：`pool_dm.py` + `scripts/` |
| `DC/` | Gradient Matching：`pool_dc.py` + `scripts/` |
| `quantize/` | 阶段二后量化（三方法共享）：`quantize_pool.py`（精简版）+ `scripts/` |
| `cross_eval/` | 阶段三多架构测评（三方法共享）：`cross_evaluate.py` + `scripts/` |
| `checkpoints/` | 预训练权重占位目录（`TM/` 与 `quantized/`，待填入或从 HuggingFace 下载） |
| `last_dm.json` | 本地实验数据快照，不建议当稳定接口依赖 |

> 注：早期 `CLAUDE.md` 提到的 `DM_new/`、`TM/encoder_v2/`、`todo.md`、各方法下重复的 `ts/`/`utils.py`/`networks.py`、`TM/quantize_pool.py`、`TM/cross_evaluate.py` 在整理后已不存在；均已去重到 `core/` 或拎到顶级目录。

## 代码关系

1. `core/` 是共享库，三方法共用一份 `ts/` / `utils.py` / `networks.py`。
2. `pool_tm.py` / `pool_dm.py` / `pool_dc.py` 是“蒸馏方法 + 压缩表达”结合的训练入口，顶部用 `sys.path.insert` 注入 repo root 后 `from core.` 导入。
3. `quantize/quantize_pool.py` 与 `cross_eval/cross_evaluate.py` 与蒸馏方法无关，任意方法产出的 pool / images+labels 都能用。

## 快速入口

- `TM/scripts/run_pool_tm.sh`
- `DM/scripts/run_pool_dm.sh`
- `DC/scripts/run_pool_dc.sh`
- `quantize/scripts/run_quantize.sh`（三方法共享）
- `cross_eval/scripts/run_cross_eval.sh`（三方法共享）