| Target | Crash points | PASS | HARD_FAIL | SILENT_DIVERGENCE | LOST_WORK | CONTROL_FAIL | Versions |
|---|---|---|---|---|---|---|---|
| accelerate Accelerator.save_state (CPU, 1 process) | 8 | 3 | 5 | 0 | 0 | 0 | - |
| torch.distributed.checkpoint (gloo, 2 ranks) | 6 | 0 | 6 | 0 | 0 | 0 | - |
| HF Trainer, GPT-2 tiny (1 process) | 12 | 2 | 10 | 0 | 0 | 0 | torch 2.14.1+cpu, transformers 5.18.0, lightning 2.6.6 |
