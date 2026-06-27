# 查找 DD-RUO 全部量化后权重的 Prompt（思维链版）

> 目标：让任意 AI agent 高效、不遗漏地定位 DD-RUO 项目中**所有量化后的权重产物**，并产出一张「方法 × 子集 × ipc → 量化前 pool / 量化后产物路径 / 完整性」的总表。

---

## 一、背景与角色

你是 DD-RUO 开源项目的代码整理助手。DD-RUO 把数据集蒸馏（TM/DM/DC 三种方法）与图像压缩（TensorPool 风格 latent + synthesis network）结合。完整 pipeline 分三阶段：

1. **阶段一**：训练合成数据集 → 产出 `pool_*.pt`（**量化前**权重）
2. **阶段二**：后量化 → 产出 `images_{mse}.pt` + `labels_{mse}.pt` + `pool.pt`（**量化后**权重，本任务目标）
3. **阶段三**：多架构测评

本任务只关心**阶段二的量化后产物**，但要顺带记录其对应的阶段一量化前 pool，以便核对来源。

## 二、已知锚点（事实，无需再验证）

1. **所有量化脚本统一命名为 `run_quantize_net*.sh`**（可能有后缀，如 `_qat`、`_2`、`_net_2`）。这是最高效的检索入口。
2. **量化前权重基本都聚集在** `/data1/home/ypliu/Final_results/`（按 TM/DM/DC 分子目录）。
3. 每个量化脚本里有两个关键变量：
   - `pool_path=...` —— 输入，指向量化前的 pool checkpoint（通常落在 `Final_results/`）
   - 保存路径变量（可能叫 `save_path` / `save_dir` / `output_dir` / `res_dir` / 脚本里 `mkdir` 的目录，或拼在 `TEST_*` 目录名里）—— 输出，量化后产物的落盘位置
4. **量化后产物按脚本的 save_path 聚集**，命名规律：`images_{0,1}_{mse}.pt` + `labels_{0,1}_{mse}.pt` + `pool_{0,1}.pt`，`mse` 取多个值（如 0.5, 0.05, 0.005, 5e-4, 5e-5, 5e-7）。
5. **部分量化在其他服务器上**（如日志服务器 `ssh -p 9973 ypliu@10.249.185.16`），思路与本机一致，需通过 ssh 执行同样的检索。

## 三、思维链（执行步骤）

### Step 0 — 明确"量化后权重"的定义
- 认定标准：目录下同时存在 `images_*.pt` 与 `labels_*.pt`（可选 `pool_*.pt`）。
- 仅 `pool_*.pt` 而无 images/labels 的，归为「量化前」，不计入本任务，但要记录其路径供溯源。

### Step 1 — 全量收集量化脚本
- 本机：`find /data1/home/ypliu -name 'run_quantize_net*.sh' 2>/dev/null`
- 兼顾可能的别名/历史脚本：`find /data1/home/ypliu -name '*quantize_net*.sh' 2>/dev/null`
- 对每个脚本，记录绝对路径。

### Step 2 — 从每个脚本提取两个锚点
对每个脚本读取内容，提取：
- `pool_path`（量化前输入）
- 保存目录变量（grep `save_path|save_dir|output_dir|res_dir|mkdir|TEST_`）
- 顺带提取：方法（TM/DM/DC，可从路径或 pool_path 推断）、子集（imagenette/imagewoof/imagemeow/imagefruit/imagesquawk/imageyellow）、ipc/dipc、mse 列表。

### Step 3 — 验证保存目录确实含量化产物
对每个 save_path：
- `ls` 检查是否存在 `images_*.pt` + `labels_*.pt`。
- 不存在的 → 标记「脚本存在但产物缺失（可能没跑完 / 在别的服务器）」。
- save_path 用了变量未展开的 → 用脚本默认值或上下文还原。

### Step 4 — 反向核对 pool_path
- 确认 `pool_path` 指向的文件真实存在，且落在 `Final_results/`（多数情况）。
- 不在 `Final_results/` 的，单独列出（可能是临时/历史路径）。

### Step 5 — 聚合去重，建总表
按 `(method, subset, ipc)` 维度聚合，一个组合可能有多条（不同 mse / 不同版本）。去重时以**保存目录 + mse 集合**为粒度。

### Step 6 — 找缺口
- 阶段一 pool 全集（`Final_results/` 下 TM/DM/DC 各子集）vs 有量化产物的组合 → 列出**缺量化产物**的 (method, subset, ipc)。
- 这些缺口可能在其他服务器，进入 Step 7。

### Step 7 — 远程服务器复扫
对每台已知服务器（如 `10.249.185.16:9973`），通过 ssh 执行 Step 1–4 的等价命令：
```
ssh -p 9973 ypliu@10.249.185.16 "find ~ -name 'run_quantize_net*.sh' 2>/dev/null"
```
思路完全一致，结果合并进总表并标注来源主机。

## 四、输出格式（必须按此输出）

### 表 1：量化后产物总表
| method | subset | ipc/dipc | 量化前 pool_path | 量化后产物目录 | mse 集合 | 产物完整? | 来源主机 | 备注 |

### 表 2：缺口清单（有量化前 pool、无量化产物）
| method | subset | ipc/dipc | pool_path | 可能原因 |

### 表 3：脚本清单
| 脚本绝对路径 | method | subset | pool_path | save_path |

## 五、效率与防漏原则

1. **以脚本为索引，不要盲搜 `images_*.pt`**——盲搜会漏掉未运行脚本的产物、且慢。脚本驱动 → save_path 聚集，是更可靠的入口。
2. **`pool_path` 与 `save_path` 必须双向验证**：脚本声明 ≠ 产物真实存在。
3. **同一脚本可能被多次复用**（改 pool_path 跑不同子集），要看脚本调用历史/注释，不能只看一份。
4. **远程服务器思路一致**，不要因为换机器就改方法。
5. 遇到路径变量未展开、脚本含 `for` 循环批量跑多个子集时，**展开循环**后再登记。

## 六、起始已知样本（用于校准检索）

- 量化前 pool 样本：`/data1/home/ypliu/Final_results/TM/imagefruit_pool_3000.pt`
- 量化脚本样本：`/data1/home/ypliu/DSproject/FreD/TM/run_quantize_net_qat.sh`（TM，未量化版？需确认）
- 论文权重散点数据：`/data1/home/ypliu/tools/散点图/last_tm.json`
- 已知量化产物聚集区：`/data1/home/ypliu/DSproject/quantize_net/ImageNet/<subset>/<ipc>/TEST_*/`

---

## 七、本轮执行发现（2026-06-27，本机）

### 7.1 两条检索路径的取舍

- **路径 A：脚本反推**（Step 1–4）——本机有 212 个 `run_quantize_net*.sh`，但大量是 QAT 历史副本，且脚本常被拷进 `TEST_*` 目录后 `save_dir` 仍指向别处，**脚本所在目录 ≠ 产物目录**，反推易错。
- **路径 B：结果 json 直查**（捷径，更可靠）——`last_tm.json` / `last_dm.json` 这类记录文件里 `file_path` 直接指向产物目录的 `log.txt`，mse/mean/bpp 都在。**优先用 json，脚本反推只作交叉验证。**

### 7.2 量化产物的真实结构（关键，与早期认知不同）

1. 量化后产物目录里**只有 `pool_*.pt` 是必需的**，`images_*.pt`/`labels_*.pt` 可由 pool 重新生成（`quantize_pool.py` 里 `TensorPool.load_slice_pool()+get_data()`）。因此**完整性只看 pool 文件，不看 images/labels**。
2. 量化后产物目录里通常有 **3 个 pool**：
   - `pool_0.pt` = **量化前** pool（baseline，等同 `Final_results/` 里那份）
   - `pool_1.pt` = **量化后** pool（**这就是要发布的量化后权重**）
   - `pool_best.pt` = 训练过程中的 best（部分目录有）
3. **同方法不同 subset 的 pool_1.pt 字节数相同**（因 arm/dim/layers_v 结构一致 → 张量形状一致），这是**正常的**，已用 md5 验证内容确实不同，非误链。

### 7.3 本机完整性矩阵（`/data1/home/ypliu/DSproject/quantize_net/`）

`pool_0.pt` + `pool_1.pt` 均在记 ✓，缺记 ✗：

| subset | TM (ipc=102) | DM (ipc=96) | DC (ipc=96) |
|---|---|---|---|
| imagenette | ✓ `ImageNet/imagenette/102/TEST_*_save_img` | ✓ `DM/.../v3_run` | ✓ `DC/.../save_img` |
| imagewoof | ✓ `ImageNet/imagewoof/102/TEST_*_save_img` | ✓ `DM/.../save_img` | ✓ `DC/.../dsa=False_v1`（仅 pool，无 images） |
| imagemeow | ✓ `ImageNet/imagemeow/102/TEST_*_save_img` | ✓ `DM/.../save_img_v2` | ✓ `DC/.../dsa=False_v1`（远程，仅 pool） |
| imagefruit | ✓ `ImageNet/imagefruit/102/TEST_*_save_img` | ✓ `DM/.../save_img_v3` | ✓ `DC/.../dsa=False_v1`（仅 pool，无 images） |
| imagesquawk | ✓ `ImageNet/imagesquawk/102/TEST_*_save_img` | ✓ `DM/.../v2_16` | ✓ `DC/.../dsa=False_v2`（仅 pool，无 images） |
| imageyellow | ✓ `ImageNet/imageyellow/102/TEST_*_save_img` | ✓ `DM/.../test_quantize_v1`（远程） | ✓ `DC/.../dsa=False_6780`（远程） |

- TM 6 子集全齐（本机）。DM/DC 各缺的子集已在远程服务器 `10.249.189.249:9973` 找到并补齐（见 7.5）。
- TM 还有 ipc=51 的 imagenette/imagewoof/imagemeow（本机有，未纳入发布主线）。
- `last_tm.json` 里 ipc=1/8/15 的小 ipc 结果实际落在 `Final_results/test_arch/`，不在 quantize_net。
- TM/imageyellow 在 `quantize_net/ImageNet/imageyellow/102/` 和 `quantize_net/0511_publish/ImageNet/imageyellow/102/` 各有一份产物（疑似副本）。

### 7.4 已整理的发布目录（软链接，未复制）

`/data1/home/ypliu/DD-RUO/checkpoints_release/{TM,DM,DC}/{subset}/`：
- `pool_origin.pt` → 软链到本机磁盘 `pool_0.pt`（量化前）
- `pool_quantized.pt` → 软链到本机磁盘 `pool_1.pt`（量化后）
- 本机 15 组（TM6 + DM5 + DC4）均软链就绪。
- 完整「规范名 ↔ 磁盘实际路径」映射表见 `checkpoints_release/MAPPING.md`。

### 7.5 远程服务器补扫结果（已完成）

远程主机 `ssh -p 9973 ypliu@10.249.189.249`，路径同样在 `/data1/home/ypliu/DSproject/quantize_net/`。3 个缺口全部找到，pool_0+pool_1 齐全：

| 缺口 | 远程目录 | pool_0 | pool_1 | images |
|---|---|---|---|---|
| DM/imageyellow | `DM/ImageNet/imageyellow/96/TEST_..._test_quantize_v1` | ✓ | ✓ | 4 |
| DC/imagemeow | `DC/ImageNet/imagemeow/96/TEST_..._dsa=False_v1` | ✓ | ✓ | 0（无 images，正常） |
| DC/imageyellow | `DC/ImageNet/imageyellow/96/TEST_..._dsa=False_6780` | ✓ | ✓ | 0（无 images，正常） |

- 按用户决定，远程这 3 组**不拷贝到本机**，仅在 `MAPPING.md` 记录远程绝对路径，发布/使用时从远程取。
- 远程复扫经验：远程 quantize_net 顶层只有 `DC/` 和 `DM/`（无 `ImageNet/` 顶层 TM 目录），DC 用 `dsa=False_*` 目录（判据必须用 pool_1.pt 而非 images，否则会漏）。
- **至此 6 subset × 3 method 全部补齐**（TM 本机 6/6，DM 本机 5 + 远程 1 = 6/6，DC 本机 4 + 远程 2 = 6/6）。

### 7.6 结果记录 json 清单（本机已知）

| json | 内容 | 方法覆盖 |
|---|---|---|
| `tools/散点图/last_tm.json` | TM 6 子集 × ipc={1,8,15,...} 的 file_path/mse/mean/bpp | TM |
| `DD-RUO/last_dm.json` | DM 5 子集 ipc=96（缺 imageyellow） | DM |
| `tools/dis_v3.json` | 按 mse×ipc 汇总的 mean/bpp/distribution | 汇总 |
| `tools/verified_count/output.json` | bits/KB/final_bpp 按 ipc | 体积统计 |
| `tools/cifar10/cifar10.json`、`cifar100.json` | CIFAR 子集（非 ImageNet 主线） | CIFAR |

> **待找**：DC 的结果记录 json（暂未找到 `last_dc.json`，DC 缺口可能正因记录未汇总）。
