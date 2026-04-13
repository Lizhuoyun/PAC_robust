# ScienceQA Mechanism Metrics

## Table

| Method | Eval | Accuracy | VWR_gamma | sigma_max | avg_gate | fragile_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LoRA | clean | 0.4831 | 369.3815 | 334.8411 | 0.3293 | 0.2440 |
| LoRA | sa | 0.4881 | 434.9686 | 404.7259 | 0.3228 | 0.1917 |
| LoRA | pa | 0.5286 | 374.6303 | 328.7051 | 0.3070 | 0.2257 |
| Plugin q10 | clean | 0.4086 | 413.7442 | 315.1554 | 0.2943 | 0.0483 |
| Plugin q10 | sa | 0.4011 | 504.5593 | 406.1967 | 0.2977 | 0.0703 |
| Plugin q10 | pa | 0.5289 | 325.3629 | 260.3724 | 0.2565 | 0.0792 |
| Plugin q25 | clean | 0.4082 | 453.8860 | 358.5828 | 0.3520 | 0.2096 |
| Plugin q25 | sa | 0.4306 | 515.6351 | 461.9331 | 0.3373 | 0.2249 |
| Plugin q25 | pa | 0.5277 | 321.5636 | 284.7055 | 0.3013 | 0.1900 |
| Plugin q50 | clean | 0.4544 | 408.9265 | 370.4587 | 0.4026 | 0.3584 |
| Plugin q50 | sa | 0.4919 | 486.1091 | 435.2074 | 0.3905 | 0.3242 |
| Plugin q50 | pa | 0.5357 | 392.9529 | 343.6067 | 0.3580 | 0.3174 |

## Readout

- `clean` 下，三个 plugin 的 `VWR_gamma` 都高于 LoRA (`369.3815`)，说明当前设定没有带来更好的 clean 机制指标。
- `clean sigma_max` 只有 `Plugin q10` (`315.1554`) 低于 LoRA (`334.8411`)；`q25/q50` 反而更高。
- `SA` 下，所有 plugin 的 `VWR_gamma` 和 `sigma_max` 都比 LoRA 更差，因此目前没有看到 plugin 对 SA 机制指标的稳健收益。
- `PA` 下，`Plugin q10/q25` 的 `VWR_gamma` 分别降到 `325.3629` / `321.5636`，低于 LoRA 的 `374.6303`；这说明 plugin 在 PA 上有局部机制优势，但没有稳定转化成 worst-class 提升。
- `Plugin q50` 在 `PA` 的 `VWR_gamma` (`392.9529`) 和 `sigma_max` (`343.6067`) 都没有优于 LoRA，说明更大的 gamma 并不占优。
