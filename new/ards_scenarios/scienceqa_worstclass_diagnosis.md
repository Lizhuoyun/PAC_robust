# ScienceQA Worst-Class Diagnosis

## Metric Definition

- `Worst-cls(clean)` = clean split 上 A/B/C/D 四类准确率的最小值，越高越好。
- `Worst-cls(SA)` = Symbol Attack 后四类准确率的最小值，越高越好。
- `Worst-cls(PA)` = Position Attack 后四类准确率的最小值，越高越好。
- 本次 `PA` 已使用修正后的扰动逻辑，因此不再与 `clean` 机械相等。

## Worst-Class Table

| Method | Worst-cls(clean) | Worst-cls(SA) | Worst-cls(PA) |
| --- | ---: | ---: | ---: |
| LoRA | 0.2031 | 0.0000 | 0.1277 |
| Plugin q10 | 0.1111 | 0.0077 | 0.1362 |
| Plugin q25 | 0.0575 | 0.0000 | 0.1277 |
| Plugin q50 | 0.1034 | 0.0000 | 0.1149 |

## Readout

- `clean worst-class` 最好的是 LoRA (`0.2031`)；三个 plugin 版本都更低，说明当前 plugin 没有稳定改善 clean 最弱类。
- `SA worst-class` 只有 `Plugin q10` 从 LoRA 的 `0.0000` 微升到 `0.0077`，`q25/q50` 仍为 `0.0000`，收益不稳定。
- `PA worst-class` 上 `Plugin q10` (`0.1362`) 略高于 LoRA (`0.1277`)，`q25` 基本持平，`q50` 更低，因此 plugin 对 PA 最弱类的价值也是 mixed。

## Per-Class Accuracy (Clean)

| Method | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| LoRA | 0.6593 | 0.4393 | 0.2196 | 0.2031 |
| Plugin q10 | 0.6534 | 0.2433 | 0.3036 | 0.1111 |
| Plugin q25 | 0.6136 | 0.2830 | 0.3259 | 0.0575 |
| Plugin q50 | 0.5626 | 0.4748 | 0.2350 | 0.1034 |

## Per-Class Accuracy (SA)

| Method | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| LoRA | 0.9292 | 0.2848 | 0.0034 | 0.0000 |
| Plugin q10 | 0.9362 | 0.0527 | 0.0189 | 0.0077 |
| Plugin q25 | 0.9415 | 0.1273 | 0.0051 | 0.0000 |
| Plugin q50 | 0.9186 | 0.3037 | 0.0069 | 0.0000 |

## Per-Class Accuracy (PA)

| Method | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| LoRA | 0.7389 | 0.4709 | 0.2546 | 0.1277 |
| Plugin q10 | 0.7472 | 0.4593 | 0.2631 | 0.1362 |
| Plugin q25 | 0.6893 | 0.5012 | 0.3019 | 0.1277 |
| Plugin q50 | 0.7460 | 0.4843 | 0.2513 | 0.1149 |

## Class-Level Pattern

- `clean` 与 `PA` 下，所有方法的最弱类都集中在 `D`；plugin 主要没有修复这个瓶颈。
- `SA` 下，所有方法都出现强烈的 `A` 偏置，而 `C/D` 接近崩塌；plugin 没有扭转这一点。
- `Plugin q25` 在 `PA` 的 `B/C` 类准确率上比 LoRA 更高，但因为 `D` 仍卡在同一水平，所以 worst-class 没有同步改善。
