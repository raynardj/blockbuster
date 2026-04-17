"""
We compose and tried this part at `notebooks/vanilla-model.ipynb`
"""

from torch import nn
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

    # some simple test running
    from blockbuster.models.vanillas import (
        BaselineGPT,
        TransformerBlock,
        CausalSelfAttention,
    )
    from blockbuster.models.config import TrainConfig

    cfg = TrainConfig()

    hook_one_off = ModelHookOneOff(
        model,
        target_modules=[
            BaselineGPT,
            TransformerBlock,
            CausalSelfAttention,
        ],
    )

    model = BaselineGPT(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.block_size,
        hidden_size=cfg.hidden_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
    )
