"""
We compose and tried this part at `notebooks/vanilla-model.ipynb`
"""

import torch
from torch import nn
from time import time
from dataclasses import dataclass, field


def get_param_count(module: nn.Module) -> int:
    """
    Get parameter count of a module
    """
    return sum(p.nelement() for p in module.parameters())


def large_int(x: int) -> str:
    """
    Format an integer as a string with commas to separate every 000
    """
    return f"{x:,}"


@dataclass
class TensorStats:
    numel: int = 0
    mean: float = 0.0
    max: float = 0.0
    min: float = 0.0
    std: float = 0.0

    def dict(self):
        """
        Convert to a dictionary
        """
        return {
            "numel": self.numel,
            "mean": self.mean,
            "max": self.max,
            "min": self.min,
            "std": self.std,
        }


def extract_activation(activation, prefix: str = "") -> dict[str, dict]:
    """
    Print out interesting information from activations
    """
    results: dict[str, TensorStats] = {}
    if isinstance(activation, torch.Tensor):
        t = activation.float()
        results[prefix] = TensorStats(
            numel=activation.numel(),
            mean=t.mean().item(),
            max=t.max().item(),
            min=t.min().item(),
            std=t.std().item() if t.numel() > 1 else 0.0,
        ).dict()
        return results
    if hasattr(activation, "items"):
        for key, value in activation.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            results.update(extract_activation(value, child_prefix))
    if isinstance(activation, (list, tuple)):
        for i, item in enumerate(activation):
            child_prefix = f"{prefix}.{i}" if prefix else str(i)
            results.update(extract_activation(item, child_prefix))
    return results


class ModelHookOneOff:
    """
    Any hooks we place, will run only once, afterward the hook will be removed.

    hook_one_off = ModelHookOneOff(
    model, target_modules=[
        BaselineGPT, TransformerBlock, CausalSelfAttention,
        ]
    )
    """

    def __init__(
        self,
        model: nn.Module,
        target_modules: list | None = None,
    ):
        self.model = model
        self.target_modules = target_modules
        self.metrics = {}

    def _reset_metrics(self):
        self.metrics = {}

    def _register_fwd_hook(self, name, module: nn.Module):
        module_name = module.__class__.__name__
        forward_clean_up_list: list = []
        m = f"{module_name}|{name}|fwd"
        self.metrics[m] = dict()

        def forward_pre_hook(module, inputs):
            with torch.no_grad():
                self.metrics[m]["fwd_input_stats"] = extract_activation(inputs)
                self.metrics[m]["fwd_start_ts"] = time()

        def forward_after_hook(module, inputs, output):
            with torch.no_grad():
                self.metrics[m]["fwd_end_ts"] = time()
                self.metrics[m]["fwd_duration"] = self.metrics[m]["fwd_end_ts"] - self.metrics[m]["fwd_start_ts"]
                self.metrics[m]["fwd_output_stats"] = extract_activation(output)

            # REMOVE forward hooks ❌🪝
            for hook in forward_clean_up_list:
                hook.remove()

        # Arm forward hooks 🔫🪝
        forward_clean_up_list.append(module.register_forward_pre_hook(forward_pre_hook))
        forward_clean_up_list.append(module.register_forward_hook(forward_after_hook))

    def wire_forward(
        self,
    ):
        """
        Wire forward hooks to the model.
        """

        for name, module in self.model.named_modules():
            module_name = module.__class__.__name__
            if self.target_modules is not None and not isinstance(module, tuple(self.target_modules)):
                continue

            self._register_fwd_hook(name, module)

    def _register_bwd_hook(self, name, module: nn.Module):
        module_name = module.__class__.__name__
        backward_clean_up_list: list = []

        m = f"{module_name}|{name}|bwd"
        self.metrics[m] = dict()

        def backward_pre_hook(module, grad_output):
            with torch.no_grad():
                self.metrics[m]["bwd_grad_output_pre_stats"] = extract_activation(grad_output)
                self.metrics[m]["bwd_start_ts"] = time()

        def backward_after_hook(module, grad_input, grad_output):
            with torch.no_grad():
                self.metrics[m]["bwd_end_ts"] = time()
                self.metrics[m]["bwd_duration"] = self.metrics[m]["bwd_end_ts"] - self.metrics[m]["bwd_start_ts"]
                self.metrics[m]["bwd_grad_output_after_stats"] = extract_activation(grad_output)
                self.metrics[m]["bwd_grad_input_after_stats"] = extract_activation(grad_input)

            # REMOVE backward hooks ❌🪝
            for hook in backward_clean_up_list:
                hook.remove()

        # Arm backward hooks 🔫🪝
        backward_clean_up_list.append(module.register_backward_hook(backward_pre_hook))
        backward_clean_up_list.append(module.register_full_backward_hook(backward_after_hook))

    def wire_backward(
        self,
    ):
        """
        Wire backward hooks to the model.
        """

        for name, module in self.model.named_modules():
            module_name = module.__class__.__name__
            if self.target_modules is not None and module_name not in self.target_modules:
                continue

            self._register_bwd_hook(name, module)


if __name__ == "__main__":

    from blockbuster.models.vanillas import (
        BaselineGPT,
        TransformerBlock,
        CausalSelfAttention,
    )
    from blockbuster.models.config import TrainConfig

    cfg = TrainConfig()

    model = BaselineGPT(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.block_size,
        hidden_size=cfg.hidden_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
    )

    hook_one_off = ModelHookOneOff(
        model,
        target_modules=[
            BaselineGPT,
            TransformerBlock,
            CausalSelfAttention,
        ],
    )

    BATCH, SEQ_LEN = 2, 12
    x = torch.randint(0, cfg.vocab_size, (BATCH, SEQ_LEN))

    hook_one_off.wire_forward()
    y_ = model(x)

    # ── expected forward metric keys ──
    # 1 BaselineGPT + 12 TransformerBlock + 12 CausalSelfAttention = 25
    fwd_keys = [k for k in hook_one_off.metrics if k.endswith("|fwd")]
    assert len(fwd_keys) == 1 + cfg.n_layer + cfg.n_layer, (
        f"expected {1 + 2 * cfg.n_layer} fwd metrics, got {len(fwd_keys)}"
    )

    # ── verify each module type appears with correct count ──
    fwd_baseline = [k for k in fwd_keys if k.startswith("BaselineGPT|")]
    fwd_blocks = [k for k in fwd_keys if k.startswith("TransformerBlock|")]
    fwd_attn = [k for k in fwd_keys if k.startswith("CausalSelfAttention|")]
    assert len(fwd_baseline) == 1
    assert len(fwd_blocks) == cfg.n_layer
    assert len(fwd_attn) == cfg.n_layer

    TENSOR_STAT_KEYS = {"numel", "mean", "max", "min", "std"}
    FWD_METRIC_KEYS = {"fwd_input_stats", "fwd_start_ts", "fwd_end_ts", "fwd_duration", "fwd_output_stats"}

    for key in fwd_keys:
        m = hook_one_off.metrics[key]

        # every forward metric must contain the full set of sub-keys
        assert FWD_METRIC_KEYS.issubset(m.keys()), f"{key} missing keys: {FWD_METRIC_KEYS - m.keys()}"

        # timing sanity
        assert m["fwd_duration"] > 0, f"{key} duration should be positive"
        assert m["fwd_end_ts"] > m["fwd_start_ts"], f"{key} end_ts should be after start_ts"

        # fwd_input_stats should always have at least one entry
        assert len(m["fwd_input_stats"]) > 0, f"{key} should have input stats"

        # validate TensorStats structure inside input stats
        for stat_key, stat in m["fwd_input_stats"].items():
            assert set(stat.keys()) == TENSOR_STAT_KEYS, (
                f"{key} input stat '{stat_key}' has wrong keys: {stat.keys()}"
            )
            assert stat["numel"] > 0, f"{key} input stat '{stat_key}' numel should be positive"
            assert stat["min"] <= stat["mean"] <= stat["max"], (
                f"{key} input stat '{stat_key}' ordering: min <= mean <= max violated"
            )
            assert stat["std"] >= 0, f"{key} input stat '{stat_key}' std should be non-negative"

        # validate TensorStats structure inside output stats (if present)
        for stat_key, stat in m["fwd_output_stats"].items():
            assert set(stat.keys()) == TENSOR_STAT_KEYS, (
                f"{key} output stat '{stat_key}' has wrong keys: {stat.keys()}"
            )
            assert stat["numel"] > 0
            assert stat["min"] <= stat["mean"] <= stat["max"]
            assert stat["std"] >= 0

    # ── per-block input numel should match (batch * seq_len * hidden_size) ──
    expected_numel = BATCH * SEQ_LEN * cfg.hidden_size
    for key in fwd_blocks:
        m = hook_one_off.metrics[key]
        input_numel = list(m["fwd_input_stats"].values())[0]["numel"]
        assert input_numel == expected_numel, (
            f"{key} input numel {input_numel} != expected {expected_numel}"
        )
        output_numel = list(m["fwd_output_stats"].values())[0]["numel"]
        assert output_numel == expected_numel, (
            f"{key} output numel {output_numel} != expected {expected_numel}"
        )

    # ── CausalSelfAttention input numel should also match ──
    for key in fwd_attn:
        m = hook_one_off.metrics[key]
        input_numel = list(m["fwd_input_stats"].values())[0]["numel"]
        assert input_numel == expected_numel, (
            f"{key} input numel {input_numel} != expected {expected_numel}"
        )
        output_numel = list(m["fwd_output_stats"].values())[0]["numel"]
        assert output_numel == expected_numel, (
            f"{key} output numel {output_numel} != expected {expected_numel}"
        )

    # ── BaselineGPT input is token ids: numel = batch * seq_len ──
    baseline_key = fwd_baseline[0]
    baseline_input_numel = list(hook_one_off.metrics[baseline_key]["fwd_input_stats"].values())[0]["numel"]
    assert baseline_input_numel == BATCH * SEQ_LEN, (
        f"BaselineGPT input numel {baseline_input_numel} != expected {BATCH * SEQ_LEN}"
    )

    # ── one-off: reset metrics and re-run, hooks should NOT fire again ──
    hook_one_off._reset_metrics()
    y_ = model(x)
    y_.logits.mean().backward()
    assert len(hook_one_off.metrics) == 0, "hooks should be one-off and not fire again"

    print("all assertions passed ✅")
