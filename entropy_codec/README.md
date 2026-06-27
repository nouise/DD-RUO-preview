# entropy_codec — B 方案：ARM 因果 + constriction 流式 CABAC

可选工具：把 TensorPool 的 `pool.pt` 权重熵编码为紧凑二进制码流（实测 ~122× 压缩，码流无损）。
把 v1 (C 方案) 的 latent 编码替换为论文级 ARM 流式编码。
**网络权重保持不变**（仍用 Laplace(0, σ_MLE) + torchac），只动 latent 部分。

## 整合说明

本目录源自早期的 `TM/encoder_v2/`，已整合为顶级可选工具目录，与主 pipeline 解耦：

- 依赖主仓库的 `core/` 共享库（`from core.ts.xxx`）；`core/__init__.py` 的 `ts`→`core.ts` 兼容 shim 保证旧 `from ts.xxx` 代码与重构前 checkpoint 仍可用。
- 入口脚本走 `core.ts.core.quantizemodel.quantize_model_no_ref_v2`（主 pipeline 也用同一函数）。
- 通过 `scripts/run_encode.sh` / `run_decode.sh` / `run_analyze.sh` 配置参数，Python 端用 argparse 覆盖默认常量。

## 依赖环境

需要 `torchac` + `constriction` 两个熵编码库（主 pipeline 不需要，仅本工具需要）。
推荐用 **conda 环境 `c3_2`**（Python 3.10 / torch 2.6.0+cu124，已预装这两个库）：

```bash
conda activate c3_2   # 或用其 python 绝对路径
```

> 注：`sre2l` 环境没有这两个库，请勿在此环境运行；`c3` 的 torch 已损坏（用 `c3_2` 替代）。

## 目录结构

```
entropy_codec/
├── encode_v2.py     编码：pool.pt → bitstream_out/（B 方案 latent + C 方案 nn）
├── decode_v2.py     解码：bitstream_out/ → decoded_out/（优化版，与 encode_v2 浮点路径一致）
├── analyze.py       分析：参数 diff + PSNR + 3×3 对比图 + ARM 估计 bpp 对比
├── codec_io.py      header.bin（schema 与 v1 不同）+ .bin 容器
├── scripts/         run_encode.sh / run_decode.sh / run_analyze.sh 包装脚本
├── README.md        本文档
├── bitstream_out/   编码产出（13 个文件，gitignored）
├── decoded_out/     解码产出（decoded.pt + 51 张 PNG + grid.png，gitignored）
└── analysis_out/    分析报告（compare_3x3.png + summary.txt，gitignored）
```

> encode_v2/decode_v2 是配套的优化版（预计算 context lookup + 逐像素单向量 ARM，编解码浮点路径完全一致）。原版 encode.py/decode.py 未纳入（解码慢 ~30 分钟）。

---

## 与 v1 的核心差异

| | v1 (encoder/, C 方案) | v2 (encoder_v2/, B 方案) |
|---|---|---|
| latent 编码 | 每 grid level 一个 Laplace(μ_l, σ_l) → torchac 一次性 | 每位置 ARM 算 (μ_i, σ_i) → constriction 流式 |
| 网络权重 | Laplace(0, σ_MLE) + torchac | **不变** |
| header magic | "TMC1" | "TMC2" |
| header 内容 | 含每 level 的 (n, lo, hi, mu, sigma) | 删除上述，加 `mask_size: uint8` |
| 解码方式 | 一次性整段 | **逐图 × 逐像素** 流式 |
| latent bpp（实测） | 2.169 | **0.864** |
| 总 bpp（实测） | 2.378 | **1.065** |
| 解码速度 | <1 秒 | ~30 分钟（51 图）|

---

## 思路

### B 方案为什么能压得更紧

ARM 是**因果**的：每个像素位置 i 的 (μ_i, scale_i) 来自它**左上方**的邻居。这给了一个比"per-level 单一 Laplace"准得多的概率模型。

C 方案不用 ARM，每层用全局统计量 (μ_l, σ_l) → bpp ~1.91；
B 方案逐位置 ARM → bpp ~0.86，与 quantize_net 的 ARM 估计对齐。

### 流式 CABAC 怎么实现

- **encoder 端**：sent 已知，一次性 ARM forward 拿全部 (μ, σ)，constriction `RangeEncoder.encode()` 一次喂入。
- **decoder 端**：拿不到 sent。**必须逐像素**：
  1. 从 sent_buffer（初始全零）取**已解码部分**作为 ARM 上下文
  2. ARM forward 拿当前位置的 (μ_i, σ_i)
  3. `RangeDecoder.decode()` 解一个 symbol，写入 sent_buffer
  4. 循环

ARM 是因果的（mask 只看左/上方），所以 decoder 用"已解码 sent"和 encoder 用"完整 sent"对每个位置 i 算出来的 (μ_i, σ_i) 完全一致。

### 51 张图怎么处理

每个 grid level 一个 RangeCoder 流，**51 张图按顺序连续编入同一个流**：
- encoder：`for img_idx in range(51): encoder.encode(sent_2d[img_idx], ...)` → 一次 `get_compressed()`
- decoder：`for img_idx in range(51): for pixel: decoder.decode(...)` → 同一个 RangeDecoder 顺序解完

代价：必须 51 张连续编/解，无法跳取单张。
好处：51 张共享流头尾开销，码率紧凑。

---

## 输入 / 输出

### encode.py

**输入**（脚本顶部常量）：
- `POOL_PATH` —— pool .pt 文件
- `SLICE_INDEX = 0` —— 取哪个 slice
- `LAYERS_V="v5", ARM_DIM=32, N_HIDDEN_ARM=4, MSE_ERR=5e-7` —— 必须与 pool 训练参数一致
- `MASK_SIZE = 9`（**关键！** TM 项目里是 9，DM_new 是 13，二者不同）
- `LAPLACE_RANGE = 60` —— constriction QuantizedLaplace 的整数支撑区间

**输出** → `bitstream_out/`（13 个文件，~109 KB）：

```
header.bin                       340 B   元信息（不再含 per-level Laplace 参数）
latent_l0.bin                 59,060 B   level 0 (51×1×128×128)，全部 51 张连续装一个 RangeCoder
latent_l1.bin                 17,724 B   level 1
latent_l2.bin                  8,180 B
latent_l3.bin                  3,768 B
latent_l4.bin                  1,184 B
latent_l5.bin                    372 B
nn_arm_weight.bin              3,720 B   网络部分仍走 v1 的 Laplace + torchac
nn_arm_bias.bin                   92 B
nn_upsampling_weight.bin          87 B
nn_upsampling_bias.bin             4 B
nn_synthesis_weight.bin       16,308 B
nn_synthesis_bias.bin            383 B
```

### decode.py

**输入**：`bitstream_out/` 全部 13 个 .bin（仅依赖于此）

**配置常量**：
- `LAPLACE_RANGE = 60` —— 必须与 encode 端一致
- `DECODE_ONLY_FIRST = False` —— 调试时设 True，只解 batch[0]（剩下 50 张为零，重建图不可用）

**输出** → `decoded_out/`：
```
decoded.pt                                4.32 MB   {'param': DPParams}
decoded_images/img_00.png ... img_50.png   每张 ~10 KB
decoded_images/grid.png                                  概览
```

### analyze.py

**输入**：原 pool + `bitstream_out/` + `decoded_out/decoded.pt`

**输出** → `analysis_out/`：
```
compare_3x3.png    3 行 × 3 列 对比图
summary.txt        bpp / 压缩率 / PSNR / 张量 max_abs_diff / 文件清单 / ARM 估计 vs 实际
```

---

## 用法

推荐用 `scripts/` 下的 sh 包装脚本（在脚本顶部修改 `POOL_PATH` 等配置）：

```bash
conda activate c3_2
cd /path/to/DD-RUO/entropy_codec

# 1. 编码（quantize_net ~13 秒 + 逐像素流式编码 latent ~5 分钟）
bash scripts/run_encode.sh      # → bitstream_out/（13 个 .bin）

# 2. 解码（逐像素流式，51 张 128×128 约 30+ 分钟；调试可设 N_IMAGES=2）
bash scripts/run_decode.sh      # → decoded_out/decoded.pt + 合成图 PNG

# 3. 分析（bpp / PSNR / max_abs_diff / 3×3 对比图，~10 秒；缺 decoded.pt 时自动解码）
bash scripts/run_analyze.sh     # → analysis_out/
```

也可直接调 Python（参数走 argparse，覆盖脚本内默认常量）：

```bash
python encode_v2.py --pool_path /path/to/pool.pt --layers_v v5 --arm_dim 32 --mask_size 9
python decode_v2.py --bitstream_dir ./bitstream_out --n_images 2   # 调试只解前 2 张
python analyze.py  --pool_path /path/to/pool.pt --bitstream_dir ./bitstream_out
```

---

## 关键数字（slice 0_0，meow，51 张 128×128）

| 项 | 值 |
|---|---|
| pool.pt 单 slice | 12.96 MB |
| **总码流** | **108.62 KB** |
| **压缩率** | **122×**（节省 99.18%） |
| latent bpp 实测 | 0.865 |
| latent bpp ARM 估计（理论）| 0.864 |
| 总 bpp 实测 | 1.065 |
| 总 bpp ARM 估计（quantize_net） | 1.048 |
| 实际/估计 比率 | **1.016**（接近最优） |
| PSNR(pre vs post) | ∞ dB（码流无损） |
| 参数 max_abs_diff | 0 |

---

## 重要踩坑

### 1. `mask_size` 必须是 9（不是 13）

DM_new 项目里 `max_mask_size=13`，TM 项目改回了 9（见 `ts/tensor_data_func_v6.py:83`）。
**用错会让 bpp 暴涨 2.5×**（因为 ARM 输入维度对不上，mu/scale 完全偏离）。

### 2. `LAPLACE_RANGE = 60` 要校验

实测 sent 范围 [-10, 11]，60 足够。但若换 pool 必须先验证 `sent.abs().max() < LAPLACE_RANGE`。
encode.py 里有 assert，触发就调大常量。

### 3. `scale_floor = 1e-6`

constriction 不接受 scale = 0。ARM 输出虽然 clamp 过，但保险起见再 clamp 一次。

### 4. CPU vs GPU 浮点行为

`Arm` 是 nn.Module，在 CPU 和 GPU 上 forward 的浮点结果**完全一致**（实测 bpp=0.9446 两端相同）。
但 ARM 参数所在 device 必须与输入 context 一致——encoder 端把 `dp.pool["ap"]` 显式 `.cpu()` 后再喂。

### 5. 解码慢

L0 是 128×128 = 16384 像素，每像素跑一次 ARM forward（5 层 MLP），总耗时 ~30 秒/图 × 51 = ~25 分钟。
当前没优化（plan 里说速度先不管）。后续可以：
- 增量更新 ARM 输入（每解一个只更新 _get_neighbor 受影响的几个位置）
- batch 维度并行（51 张图一起跑 ARM forward，但流式 decoder 必须逐图——这个矛盾难解）

---

## TODO

- 把 `mask_size / arm_dim / n_hidden_arm / layers_v` 写到比特流（v2 仍硬编）
- 解码加速（增量 ARM forward）
- 多 slice 支持
