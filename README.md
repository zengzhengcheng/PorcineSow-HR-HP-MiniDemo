# PorcineSow-HR-HP-MiniDemo

[![CI](https://github.com/zengzhengcheng/PorcineSow-HR-HP-MiniDemo/actions/workflows/ci.yml/badge.svg)](https://github.com/zengzhengcheng/PorcineSow-HR-HP-MiniDemo/actions)
[![License](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/data-CC--BY--4.0-lightgrey.svg)](DATA_LICENSE)

Minimal data and baseline for predicting 5-min heat production (HP, kcal/5 min)
in pregnant sows from non-invasive heart-rate and back-mounted inertial-sensor
features. Companion to a short paper section; the broader pipeline and
out-of-distribution evaluation are reported elsewhere (see *Related projects*
below).

## What you get

- `data/sow_features_minimal.csv` - 24,066 five-minute windows from 14 sows,
  12 features + 1 target, fully anonymised.
- `code/train_baseline.py` - single-config 5-fold cross-validation training
  script using `sklearn.HistGradientBoostingRegressor`.
- `results/cv_summary.csv` - fold-by-fold metrics shipped with the repo so
  readers can verify their re-run matches.

## Reproduce in three commands

```bash
conda env create -f environment.yml
conda activate sow-mini
python code/train_baseline.py --config code/config_baseline.yaml
```

Pip-only alternative:

```bash
pip install -r requirements.txt
python code/train_baseline.py --config code/config_baseline.yaml
```

Expected output (5-fold KFold, shuffle, seed = 42):

```
5-fold R^2 : 0.81  +/- 0.01
5-fold MAE : 2.27  kcal / 5 min
```

Exact fold-level values are in `results/cv_summary.csv` and `results/cv_summary.md`.

## Data

`data/sow_features_minimal.csv` columns:

| group | columns | unit |
|---|---|---|
| identifier | `animal_id, period_in_animal, day_in_period, time_offset_s` | - |
| heart rate | `hr_mean, hr_sd` | bpm |
| heart-rate variability | `rmssd, pnn50` | ms, % |
| motion | `gyro_mean, angle_sd, dz_sd` | rad/s, rad, rad/s^2 |
| activity ratios | `active_fraction, frac_rest` | 0-1 |
| body weight | `weight_kg` | kg |
| context | `hour_of_day, is_fasting_day` | 0-23, 0/1 |
| target | `hp_kcal_5min` | kcal / 5 min |

Identifiers are anonymised: `animal_id` is `S01..S14` assigned by first
appearance order; `time_offset_s` is integer seconds from the start of each
(animal, period) block. Absolute dates and original chamber labels are not
retained. SHA-256 of the shipped CSV is in `data/sha256sums.txt`. Full column
dictionary in `data/DATA_README.md`.

The data is pre-filtered to `hr_valid == 1` and `move_coverage >= 0.5` from
an upstream feature table; those quality gates are not retained as columns
since they are constant or near-constant on the kept rows. A trim
`11 <= hp_kcal_5min <= 70` kcal / 5 min is then applied to the target to
stabilise the regression target (removes 0.30% of rows that correspond to
chamber transients on the low end and a small number of windows above the
typical HP range observed in this cohort on the high end; see
`data/DATA_README.md` for details).

## Method

`sklearn.ensemble.HistGradientBoostingRegressor` with a fixed configuration
(see `code/config_baseline.yaml`):

- `max_iter = 500`
- `learning_rate = 0.05`
- `max_leaf_nodes = 31`
- `min_samples_leaf = 20`
- `random_state = 42`

Evaluation: `KFold(n_splits=5, shuffle=True, random_state=42)`. Reported
metrics are R^2, MAE, RMSE per fold plus mean and std across folds. A
permutation-importance plot averaged across folds is written to
`results/feature_importance.png`.

## Citation

Please cite the archived Zenodo release once the DOI is assigned (badge above).
Until then, see `CITATION.cff` for full metadata.

## Related projects

This minimal demo is a companion to a software family released by the same
group. None of the code here is reused from those projects (independent
reimplementation), but readers interested in upstream signal processing should
consult:

- **SwineSync-OpenSource** - <https://github.com/zengzhengcheng/SwineSync-OpenSource> - DOI [10.5281/zenodo.20051135](https://doi.org/10.5281/zenodo.20051135) - HRV processing, motion feature aggregation, ECG-TransUNet ONNX inference.
- **OpenCalori-Swine** - <https://github.com/zengzhengcheng/OpenCalori-Swine> - DOI [10.5281/zenodo.20051163](https://doi.org/10.5281/zenodo.20051163) - Brouwer-equation heat production calculator.
- **ECG-TransUNet** - <https://github.com/zengzhengcheng/ECG-TransUNet> - DOI [10.5281/zenodo.20051167](https://doi.org/10.5281/zenodo.20051167) - ECG R-peak detection model training code.

## Authors

- Zhengcheng Zeng (first author) - `zengzhengcheng@cau.edu.cn`
- Zhenyu Lei - `leizhenyu@cau.edu.cn`
- Yuyu Gao - `gaoyuyu@cau.edu.cn`
- Shuai Zhang (corresponding) - `zhangshuai16@cau.edu.cn`

All affiliated with the College of Animal Science and Technology, China
Agricultural University, Beijing, China.

## License

- Code (`code/`, `.github/`): MIT - see `LICENSE`.
- Data (`data/`): CC BY 4.0 - see `DATA_LICENSE`.
- Documentation (`README.md`, `docs/`): CC BY 4.0.

---

## 中文说明

本仓库随同一篇关于妊娠母猪 5 分钟产热预测的短文小节发布，提供精简后的特征数据
和单一配置的基线训练脚本。完整方法学和更严格的泛化评估见对应论文。

### 数据

`data/sow_features_minimal.csv`：14 头母猪的 24,066 个 5 分钟窗口，含心率衍生
（4 列）、运动统计（5 列）、体重、时段上下文（共 12 个特征）+ 1 个目标
`hp_kcal_5min`（kcal/5 min）。窗口已按 `hr_valid == 1` 与 `move_coverage >= 0.5`
预筛过，并对目标做了范围修剪 `11 ≤ hp_kcal_5min ≤ 70` kcal/5 min 以稳定回归目标
（剔除 0.30%，低端为小室门启闭瞬变，高端为少数超出本队列典型 HP 范围的窗口，
详见 `data/DATA_README.md`）。母猪编号已匿名为 `S01..S14`，时间戳替换为相对秒，
原始小室与日期不保留。

### 快速复现

```bash
conda env create -f environment.yml
conda activate sow-mini
python code/train_baseline.py --config code/config_baseline.yaml
```

预期 5-fold 交叉验证 R² 约 0.81，MAE 约 2.27 kcal/(5 min)。

### 模型

`sklearn.ensemble.HistGradientBoostingRegressor`，固定超参（见 `code/config_baseline.yaml`），
`KFold(n_splits=5, shuffle=True, random_state=42)`。结果写入 `results/`。

### 引用

请引用 Zenodo 归档版本（DOI 待第一次 release 后回填到 `CITATION.cff`）。

### 许可

代码 MIT，数据 CC BY 4.0。

### 通讯作者

张帅 `zhangshuai16@cau.edu.cn` — 中国农业大学动物科学技术学院。
