## Metrics 📋

Metrics is to mesure the amount of change / impact happened after applying a `hack`.

### Compute & Throughput — *Speed Impact* ⚡🖥️

Each metric is a **ratio or % change** vs. the baseline (no hack). Scan the number and direction to judge impact instantly.

| Metric | Formula | Intuition |
|---|---|---|
| **TPS Speedup** 🔥⏩ | `TPS_hack / TPS_base` | `1.4x` = 40% faster. `<1x` = regression ⚠️. |
| **MFU Gain** 📊🧮 | `MFU_hack − MFU_base` (pp) | Percentage-point improvement in hardware utilization 🔌. |
| **TTFT Reduction** 🚀⏱️ | `(TTFT_base − TTFT_hack) / TTFT_base` | % latency cut in the **prefill** phase 📝. |
| **TPOT Reduction** 🔁⚡ | `(TPOT_base − TPOT_hack) / TPOT_base` | % faster per generated token 🔤. GQA / MQA shine here ✨. |
| **Backward Overhead Ratio** ↔️🔄 | `(Bwd/Fwd)_hack / (Bwd/Fwd)_base` | `>1` = hack added gradient overhead 🧩. |

### Memory & Resource — *Memory Impact* 🧠💾

Memory savings directly translate to larger **batch size** 📚 or longer **sequence length** 📐.

| Metric | Formula | Intuition |
|---|---|---|
| **VRAM Savings** 🎮📈 | `(VRAM_base − VRAM_hack) / VRAM_base` | % peak memory freed during the backward pass ⬅️. |
| **Activation Footprint Reduction** 📦🧬 | `Act_base / Act_hack` | `2x` = half the activation memory needed ⚖️. |
| **KV Cache Reduction** 🔑🗄️ | `(KV_base − KV_hack) / KV_base` | % KV memory saved vs. sequence length 📏. *(GQA/MQA vs. vanilla MHA—big wins here 🏆.)* |
| **Batch Capacity Gain** 📏🧪 | `MaxBatch_hack / MaxBatch_base` | `1.5x` = fits 50% more samples before OOM 💥. |

### Quality & Performance — *Quality Tax* ✨🎓

Hacks can dull the learning signal 🧪—these should all stay **near zero**. Negative = free lunch 🍀.

| Metric | Formula | Intuition |
|---|---|---|
| **Perplexity Tax** 📉📈 | `PPL_hack − PPL_base` | Positive = degradation. Track the full loss curve AUC 🏃, not just the endpoint. |
| **Accuracy Delta** 🎯🧩 | `Acc_hack − Acc_base` per benchmark | Negative = regression. Check MMLU, Hellaswag, PIQA slices ✅. |
| **Context Fidelity Retention** 🪡🌾 | `NiH_hack / NiH_base` at each length | Ratio of Needle-In-A-Haystack 🌿 score at 1k / 4k / 8k. `1.0` = no regression 🧭. |

### Training Dynamics — *Stability Impact* ⚖️🎢

Some hacks (skipping LayerNorm, heavy quant, …) make training a **roller coaster** 🎢. Ratios tell you how much worse or better.

| Metric | Formula | Intuition |
|---|---|---|
| **Spike Rate Ratio** ⚠️💥 | `Spikes_hack / Spikes_base` | `<1` = more stable 🟢. `>1` = hack destabilizes training. |
| **Gradient Norm Shift** 📐📊 | `median(‖g‖_hack) / median(‖g‖_base)` | Drift from `1.0` signals **exploding** 🧨 / **vanishing** 🫥 gradients. |
| **Outlier Amplification** 🌡️🔭 | `max|h|_hack / max|h|_base` | `>1` = larger activation outliers 🦒—bad news for PTQ to INT8 / FP8 🔢. |
| **Token Efficiency** ⏱️🎯 | `Tokens_base→target / Tokens_hack→target` | `>1` = hack converges faster ✅. `<1` = needs more data to match baseline ⚖️. |

### Conveniences
* 💾 **Pretrain Compatibility** : Does this hack eliminate our possibility to use model weights trained without such hack?
* **Compatibility with Other Hacks**: This should be represented by levels of compatibility -> a list of hacks, eg.:
```json
{
    "easy": [
        "hack1",
        "hack2",
    ],
    "with_code_change": [
        "hack4",
    ],
    "impossible": [
        "hack5",
    ]
}