# GPU Training Options

Free and cheap GPU compute options for running BlockBuster metric experiments.

---

## Free GPU Options

| Platform | GPU | VRAM | Limits | Notes |
|---|---|---|---|---|
| **Kaggle Notebooks** | P100 | 16GB | 30 hrs/week | Most reliable free option, background execution |
| **Google Colab** | T4 | 16GB | ~15-30 hrs/week | No guarantee, 12hr session max |
| **Paperspace Gradient** | T4/RTX4000 | varies | 6hr sessions, unlimited restarts | No weekly quota — just restart |
| **AWS SageMaker Studio Lab** | mixed | varies | Free tier | Student-focused, limited GPU |
| **HuggingFace ZeroGPU** | shared | varies | Daily quota | Better for inference than training |

**Practical free stack:** Kaggle + Colab + Paperspace = ~45-60 free GPU hours/week

---

## Cheap Paid Options

| Platform | GPU | VRAM | Price/hr |
|---|---|---|---|
| **RunPod** | RTX 3090 | 24GB | $0.22 |
| **Vast.ai** | RTX 4090 | 24GB | $0.29 |
| **RunPod** | A40 | 48GB | $0.35 |
| **Vast.ai** | A100 | 40GB | $1.29 |
| **Lambda Labs** | A100 | 40GB | $1.10 |

- **Vast.ai** — cheapest (peer-to-peer marketplace, prices fluctuate with supply/demand)
- **RunPod** — best balance of price + reliability

---

## Recommendation for BlockBuster

Since metric comparison runs between vanilla models and hacks are iterative but not massive:

1. **Dev / quick experiments** — Kaggle Notebooks (free, 30 hrs/week)
2. **Longer metric sweeps** — Paperspace Gradient free tier (6hr sessions, restart freely)
3. **Serious runs** — RunPod RTX 3090 at **$0.22/hr** (~$22 per 100 GPU hours)

### GPU Memory by Model Size

- 7B–13B params → T4/P100 (16GB) is sufficient
- 13B–30B params → RTX 3090/4090 (24GB) needed
- Rough estimate: `(params × 8 bytes × 3) / 1e9 = GB needed`

---

## Academic Grants

- **NVIDIA Academic Grant Program** — free GPU for research with publications: https://academicgrants.nvidia.com/
- **Lambda Labs** — free GPU time for academic researchers with published projects
- **HuggingFace GPU Grants** — limited availability for open-source projects

---

## References

- https://www.gmicloud.ai/blog/where-can-i-get-free-gpu-cloud-trials-in-2026-a-complete-guide
- https://iotbyhvm.ooo/best-free-cloud-gpu-platforms-in-2026-google-colab-kaggle-and-more/
- https://blog.paperspace.com/paperspace-launches-gradient-community-notebooks/
- https://vast.ai/pricing/gpu/RTX-4090
- https://www.runpod.io/pricing
- https://northflank.com/blog/cheapest-cloud-gpu-providers
