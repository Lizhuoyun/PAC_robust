# ScienceQA q25 Alpha Sweep

## Target Sweep

- Gamma fixed at `q25 = -0.42339251935482025`.
- Planned alpha values: `0.02`, `0.05`, `0.10`.
- Planned outputs: `Clean / SA / PA accuracy`, `Worst-cls(clean/SA/PA)`, `VWR_gamma`, `sigma_max`.

## What Was Prepared

- Added `--alpha` and `--output_suffix` to `sqa_plugin_trainer.py` so sweep checkpoints do not overwrite each other.
- Added optional `--batch_size` and `--grad_accum` overrides for constrained-memory retries.
- Reworked plugin training to backprop `clean CE` and `SA regularization` sequentially within the same step to reduce peak memory.
- Enabled gradient checkpointing in the LoRA training path to further reduce training memory.

## Current Status

- No alpha checkpoint has completed yet, so there is no valid sweep result table to report at this moment.
- `batch_size=4` still OOMs on a 40GB A100 during plugin training.
- `batch_size=1` can start, but became numerically unstable very early (`ce=nan` around step 16) and later also OOMed during the SA branch.
- `batch_size=2` remained memory-bound in the clean forward path.
- During later retries, all three GPUs were also occupied by other long-running jobs, preventing a clean rerun window.

## Interim Conclusion

- The corrected diagnosis now strongly suggests the next bottleneck is training feasibility, not evaluation correctness.
- Before a meaningful q25 alpha sweep can finish, the training recipe likely needs one additional stabilization pass (for example: a proven memory-safe batch configuration and a fix for early-step NaNs under micro-batching).
