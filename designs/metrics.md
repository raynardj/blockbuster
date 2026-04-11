## Metrics 📋

### Compute & Throughput Metrics (Speed) ⚡🖥️

These metrics determine how efficiently the architecture utilizes the hardware—during **training** 🏋️ and **inference** 🔮 alike.

- **Tokens Per Second (TPS)** 🔥⏩: Raw throughput. Measure during training (tokens/sec/GPU) 🖲️ and inference (generation speed) ✍️.
- **Model FLOPS Utilization (MFU) / Hardware FLOPS Utilization (HFU)** 📊🧮: Ratio of theoretical ops your model needs vs. the GPU’s theoretical max 🔌. Crucial for custom CUDA kernels or FlashAttention ⚡.
- **Time to First Token (TTFT)** 🚀⏱️: Inference-only. Latency of the **prefill** phase (prompt processing) 📝. FlashAttention-style hacks move the needle here.
- **Time Per Output Token (TPOT)** 🔁⚡: Inference-only. **Decoding** latency per generated token 🔤. GQA / MQA shine here ✨.
- **Forward vs. Backward Pass Time Ratio** ↔️🔄: Spots whether a hack bottlenecks gradients—e.g. gradient checkpointing 🧩 or custom normalizations.

### Memory & Resource Metrics (GPU Usage) 🧠💾

Memory bottlenecks often dictate max **batch size** 📚 and **sequence length** 📐.

- **Peak VRAM Usage (Training)** 🎮📈: Max memory during the **backward** pass ⬅️. Essential for gradient checkpointing 🧩 or activation offloading 📤.
- **Activation Memory Footprint** 📦🧬: Activations only—separate from **weights** ⚖️.
- **KV Cache Size (Inference)** 🔑🗄️: Memory for Keys & Values during generation 🌊. Track scaling vs. sequence length 📏. *(GQA/MQA vs. vanilla MHA—big wins here 🏆.)*
- **Batch Size Tolerance** 📏🧪: Largest batch that fits before **OOM** 💥 at a fixed sequence length.

### Quality & Performance Metrics (Efficacy) ✨🎓

Architectural hacks can dull the learning signal 🧪—measure whether the model **actually** learns 🌱.

- **Validation Perplexity / Cross-Entropy Loss** 📉📈: The usual baseline. Track **AUC** of the loss curve—not only the endpoint—to see **learning speed** 🏃.
- **Zero-shot / Few-shot Downstream Accuracy** 🎯🧩: Perplexity ≠ reasoning 🤔. Quick evals (MMLU, Hellaswag, PIQA slices, …) keep you honest ✅.
- **Effective Context Window** 🪡🌾: Long-sequence utility. **Needle In A Haystack** 🌿 or PPL at 1k / 4k / 8k tests RoPE, ALiBi, and friends 🧭.

### Training Dynamics & Stability Metrics ⚖️🎢

Some hacks (skipping LayerNorm, heavy quant, …) make training a **roller coaster** 🎢—watch these.

- **Loss Spikes / NaN Frequency** ⚠️💥: How often loss **spikes** 📈 or you need LR back-tracks ⏪.
- **Gradient Norm Magnitude & Variance** 📐📊: Watch the $L_2$ norm—**exploding** 🧨 / **vanishing** 🫥 gradients show up before NaN.
- **Activation Outlier Magnitude** 🌡️🔭: Max |hidden| values. RMSNorm / some activations spawn **outliers** 🦒—bad news for PTQ to INT8 / FP8 🔢.
- **Step-to-Convergence / Tokens-to-Target-Loss** ⏱️🎯: Steps or **M tokens** to hit a fixed val loss—the **time vs. quality** tradeoff ⚖️.
