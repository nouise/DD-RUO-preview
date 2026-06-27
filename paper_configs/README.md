# Paper configs — actual experiment launch scripts & log locations

This folder collects the **exact per-experiment launch scripts** used to produce the
results reported in the paper, plus a pointer to where each run's **logs and final
weights** live on our training servers. Scripts are copied here verbatim (symlinks
dereferenced); logs are **not** copied — their server paths are annotated below.

Three distillation methods share one TensorPool compression backbone:
**TM** (Trajectory Matching), **DM** (Distribution Matching), **DC** = **GM**
(Gradient Matching, called "GM" in the paper).

---

## 1. The two coefficients (paper essence)

Every run is governed by two rate coefficients. In code they are realized as **two
separate gradient amplifiers** (`lr_scale` on the distillation gradient, `ldb_scale`
on the rate gradient — see `run_model_backward`). Both relate to the paper symbols by
a **reciprocal**:

| Paper symbol | Phase | Code formula | Meaning |
| --- | --- | --- | --- |
| **β** | warmup / init (`init_from_data` → `run_warmup`) | `β = 1 / ldb_warmup` | rate–distortion trade-off used to build `pool_init.pt` |
| **λ** | joint rate–utility optimization | `λ = lr_it / (ldb_it × ldb)` | rate–utility trade-off during distillation |

Equivalently `λ = β_joint × (lr_it / ldb_it)` where `β_joint = 1/ldb`.

- A **larger λ** prioritizes distillation utility (looser bit-rate);
  a **smaller λ** tightens the bit-rate. The two-stage schedule **raises `ldb_it`** in
  stage 2 to shrink λ.
- The warmup `ldb` and the joint `ldb` are **different quantities**. DM/DC use joint
  `ldb=5` but must warm up **separately** with `ldb_warmup=1e-6` to reach `β=1e6`.
  DC reuses DM's warmup `pool_init.pt`.

### Per-method constants

| Method | warmup `ldb` → **β** | joint `ldb` | `lr_it` | λ = lr_it/(ldb_it·ldb) |
| --- | --- | --- | --- | --- |
| TM | 0.1 → **10** | 0.1 | 1000 | `10000 / ldb_it` |
| DM | 1e-6 → **1e6** | 5 | 1000 | `200 / ldb_it` |
| DC (GM) | 1e-6 → **1e6** | 5 | 100 | `20 / ldb_it` |

---

## 2. Server / log legend

| Server | SSH | Project root | Holds |
| --- | --- | --- | --- |
| **S-TMDM** | `ssh -p 9973 ypliu@10.249.185.16` | `/data2/home/ypliu/DSproject/` | TM (all) + DM joint stages |
| **S-DC** | `ssh -p 9973 ypliu@10.249.189.249` | `/data1/home/ypliu/DSproject/` | DC (all) + DM warmup/stage1 |

**Log location rule:** each run writes its log to
`{save_path}/{FLAG}/{timestamp}_{batch_syn}.log`, where `FLAG` encodes the params:
`ImageNet_{subset}_{ipc}ipc_ConvNetD5_{METHOD}_pool_1_{Iteration}_{ldb}_{lr_img}_{lr_it}_{ldb_it}_{res}_zca_False_#{TAG}`.
The `save_path` for each run is given in the tables below; the log + checkpoints
(`*.pt`) sit inside the matching `FLAG` sub-directory.

---

## 3. TM (Trajectory Matching) — β=10, joint ldb=0.1, lr_it=1000

`save_path = /data2/home/ypliu/DSproject/tm_result/ImageNet/{subset}/{ipc}/` on **S-TMDM**
(except where noted). λ = 10000 / `ldb_it`.

| ipc | subset | stage | script | Iter | ldb_it | **λ** | pool_init / notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 102 | imagenette | single | `TM/ipc102/imagenette/stage1.sh` | 10000 | 150 | 66.7 | `init` (auto-warmup) |
| 102 | imagefruit | 1 | `TM/ipc102/imagefruit/stage1.sh` | 7000 | 150 | 66.7 | `init` |
| 102 | imagefruit | 2 | `TM/ipc102/imagefruit/stage2.sh` | 2000 | 150 | 66.7 | from stage1 `_7000_..._150` |
| 102 | imagemeow | 1 | `TM/ipc102/imagemeow/stage1.sh` | 8000 | 10 | 1000 | `init` |
| 102 | imagemeow | 2 | `TM/ipc102/imagemeow/stage2.sh` | 7000 | 150 | 66.7 | from stage1 `_8000_..._10` |
| 102 | imagewoof | 1 | `TM/ipc102/imagewoof/stage1.sh` | 7000 | 150 | 66.7 | `init` |
| 102 | imagewoof | 2 | `TM/ipc102/imagewoof/stage2.sh` | 2000 | 150 | 66.7 | from stage1 `_7000_..._150` |
| 102 | imageyellow | 1 | `TM/ipc102/imageyellow/stage1.sh` | 8000 | 10 | 1000 | from `_1_..._10` |
| 102 | imageyellow | 2 | `TM/ipc102/imageyellow/stage2.sh` | 7000 | 150 | 66.7 | from stage1 `_8000_..._10` |
| 51 | imagenette | 1 | `TM/ipc51/imagenette_v1/stage1.sh` | 25000 | 100 | 100 | `init`; save_path `results_0124/` |
| 15 | imagenette | v1 | `TM/ipc15/imagenette_v1/stage1.sh` | 25000 | 100 | 100 | `init`; save_path `results_0119/` |
| 15 | imagenette | v2/v3 s1 | `TM/ipc15/imagenette_v2|v3/stage1.sh` | 8000 | 10 | 1000 | from `_8000_..._10` |
| 15 | imagenette | v2 s2 | `TM/ipc15/imagenette_v2/stage2.sh` | 12000 | 150 | 66.7 | from stage1 |
| 15 | imagenette | v3 s2 | `TM/ipc15/imagenette_v3/stage2.sh` | 12000 | 120 | 83.3 | from stage1 |
| 15 | imagemeow | 1 | `TM/ipc15/imagemeow_v2/stage1.sh` | 8000 | 10 | 1000 | from `_5_..._10` |
| 15 | imagemeow | 2 | `TM/ipc15/imagemeow_v2/stage2.sh` | 7000 | 120 | 83.3 | from stage1 |
| 8 | imagenette | v0 | `TM/ipc8/imagenette_v0/stage1.sh` | 25000 | 20 | 500 | `init`; save_path `results_0206/` |
| 8 | imagenette | v1 | `TM/ipc8/imagenette_v1/stage1.sh` | 25000 | 100 | 100 | `init`; save_path `results_0119/` |
| 8 | imagenette | v2 s1 | `TM/ipc8/imagenette_v2/stage1.sh` | 8000 | 10 | 1000 | from `_8000_..._10` |
| 8 | imagenette | v2 s2 | `TM/ipc8/imagenette_v2/stage2.sh` | 12000 | 150 | 66.7 | from stage1 |
| 8 | imagenette | v3 s1 | `TM/ipc8/imagenette_v3/stage1.sh` | 8000 | 10 | 1000 | from `_8000_..._10` |
| 8 | imagenette | v3 s2 | `TM/ipc8/imagenette_v3/stage2.sh` | 7000 | 120 | 83.3 | from stage1 |
| 8 | imagenette | v3_1 s1 | `TM/ipc8/imagenette_v3_1/stage1.sh` | 8000 | 10 | 1000 | from `_8000_..._10` |
| 8 | imagenette | v3_1 s2 | `TM/ipc8/imagenette_v3_1/stage2.sh` | 12000 | 150 | 66.7 | from stage1 |
| 8 | imagemeow | 1 | `TM/ipc8/imagemeow_v2/stage1.sh` | 8000 | 10 | 1000 | `init` |
| 8 | imagemeow | 2 | `TM/ipc8/imagemeow_v2/stage2.sh` | 12000 | 120 | 83.3 | from stage1 |
| 8 | imagewoof | 1 | `TM/ipc8/imagewoof_v2/stage1.sh` | 8000 | 10 | 1000 | `init` |
| 8 | imagewoof | 2 | `TM/ipc8/imagewoof_v2/stage2.sh` | 12000 | 120 | 83.3 | from stage1 |

> TM trajectory hyper-params (`syn_steps=40`, `expert_epochs=2`, `max_start_epoch=20`,
> `lr_lr=1e-5`, `lr_teacher=1e-2`) are auto-filled by `TM/hyper_params.py` `load_default()` for ImageNet.

---

## 4. DM (Distribution Matching) — β=1e6, joint ldb=5, lr_it=1000

`save_path = /data2/home/ypliu/DSproject/dm_result/ImageNet/{subset}/{ipc}/` (joint
stages on **S-TMDM**; warmup `pool_init.pt` produced on **S-DC** with `ldb_warmup=1e-6`).
λ = 200 / `ldb_it`.

| subset | stage | script | Iter | ldb_it | **λ** | pool_init |
| --- | --- | --- | --- | --- | --- | --- |
| imagenette | 1 | `DM/imagenette/stage1.sh` | 10000 | 50 | 4.0 | `nette_pool_init.pt` (warmup ldb=1e-6) |
| imagenette | 2 | `DM/imagenette/stage2.sh` | 10000 | 450 | 0.44 | from stage1 `_10000_..._50` |
| imagefruit | 2 | `DM/imagefruit/stage2.sh` | 10000 | 400 | 0.50 | `imagefruit_..._exp0_6000.pt` |
| imagewoof | 2 | `DM/imagewoof/stage2.sh` | 10000 | 300 | 0.67 | `imagewoof_..._exp0_8000.pt` |
| imagesquawk | 2 | `DM/imagesquawk/stage2.sh` | 10000 | 400 | 0.50 | `imagesquawk_..._exp0_4000.pt` |

> imagemeow DM reuses a TM-result pool (`tm_result/.../imagemeow/8/..._stage2_from0.1/pool_2000.pt`).
> The `imagefruit/woof/squawk` stage-1 pools live under `pt_files/` on **S-TMDM**;
> their warmups were run on **S-DC**.

---

## 5. DC = GM (Gradient Matching) — β=1e6, joint ldb=5, lr_it=100

All on **S-DC** (`/data1/home/ypliu/DSproject/`). λ = 20 / `ldb_it`.
`save_path = .../final_dc_result_0226/ImageNet/{subset}/96/` (imagesquawk uses older `dc_result/`).

| subset | stage | script | Iter | ldb_it | **λ** | pool_init |
| --- | --- | --- | --- | --- | --- | --- |
| imagefruit | 1 | `DC/imagefruit/stage1.sh` | 500 | 700 | 0.0286 | DM warmup pool (ldb=1e-6) |
| imagefruit | 2 | `DC/imagefruit/stage2.sh` | 500 | 1500 | 0.0133 | from stage1 `_500_..._700` |
| imagenette | 1 | `DC/imagenette/stage1.sh` | 500 | 700 | 0.0286 | DM warmup pool |
| imagenette | 2-inter | `DC/imagenette/stage2_inter.sh` | 500 | 1700 | 0.0118 | from stage1 `_700` |
| imagenette | 2 | `DC/imagenette/stage2.sh` | 500 | 2000 | 0.0100 | from stage2_inter `_1700` |
| imagewoof | 1 | `DC/imagewoof/stage1.sh` | 500 | 700 | 0.0286 | DM warmup pool |
| imagewoof | 2-inter | `DC/imagewoof/stage2_inter.sh` | 500 | 1500 | 0.0133 | from stage1 `_700` |
| imagewoof | 2 | `DC/imagewoof/stage2.sh` | 500 | 2000 | 0.0100 | from stage2_inter `_1500` |
| imagemeow | 1 | `DC/imagemeow/stage1.sh` | 500 | 700 | 0.0286 | `dc_result/.../_1000_..._700` |
| imagemeow | 2-inter | `DC/imagemeow/stage2_inter.sh` | 500 | 1700 | 0.0118 | from stage1 `_700` |
| imagemeow | 2 | `DC/imagemeow/stage2.sh` | 500 | 2000 | 0.0100 | from imagewoof stage2 `_1500` |
| imagesquawk | 1 | `DC/imagesquawk/stage1.sh` | 1000 | 700 | 0.0286 | DM warmup pool |
| imagesquawk | 2 | `DC/imagesquawk/stage2.sh` | 500 | 1500 | 0.0133 | from stage1 `_1000_..._700` |

> DC `outer_loop`/`inner_loop` are auto-set by `get_loops(ipc)` in `pool_dc.py`.
> Other DC distillation defaults: `--method=DC --lr_net=0.01 --batch_real=256 --init=real --dis_metric=ours`.

---

## 6. How to use a script

```bash
# 1) edit data_path / save_path / pool_init for your machine
# 2) launch (writes a .log into save_path/FLAG/)
bash paper_configs/scripts/TM/ipc102/imagenette/stage1.sh
```

The repo template scripts under `TM/scripts/`, `DM/scripts/`, `DC/scripts/` carry the
same params with placeholder paths and inline β/λ documentation.
