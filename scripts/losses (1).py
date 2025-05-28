import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple

# Consolidated epsilon constants - defined once to avoid dictionary lookups
EPS_TINY = 1e-6
EPS_SMALL = 1e-5
EPS_MEDIUM = 5e-3
EPS_LARGE = 5e-2


class BaseDivergence(nn.Module):
    """Base class for all divergence measures"""

    def __init__(self, nclass: int, param: Optional[List[float]] = None,
                 softmax_logits: bool = True, softmax_gt: bool = True):
        """
        Initialize base divergence measure.

        Args:
            nclass: Number of classes
            param: Optional parameters for specific divergence measures
            softmax_logits: Whether to apply softmax to model predictions
            softmax_gt: Whether to apply softmax to ground truth distributions
        """
        super(BaseDivergence, self).__init__()
        self.nclass = nclass
        self.param = [] if param is None else param
        self.softmax_logits = softmax_logits
        self.softmax_gt = softmax_gt
        assert nclass >= 2, "Number of classes must be at least 2"

    def prepare_inputs(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process inputs based on their shapes.

        Args:
            logits: Model predictions
            targets: Ground truth (class indices or distributions)

        Returns:
            Processed logits and targets
        """
        if len(targets.shape) == len(logits.shape) - 1:
            # If targets are class indices, convert to one-hot
            targets = F.one_hot(targets, self.nclass).to(logits.dtype).to(logits.device)

            # If targets are already distributions
        if self.softmax_logits:
            logits = F.softmax(logits, dim=1)

        # Only apply softmax to targets if they're not already one-hot encoded and softmax_gt is True
        if self.softmax_gt and not torch.allclose(targets.sum(dim=1),
                                                  torch.ones(targets.size(0), device=targets.device)):
            targets = F.softmax(targets, dim=1)

        return logits, targets

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """To be implemented by child classes"""
        raise NotImplementedError


class KL(BaseDivergence):
    """Kullback-Leibler Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Using clamp for numerical stability instead of adding epsilon
        log_probs = torch.log(torch.clamp(logits, min=EPS_TINY))
        return (-targets * log_probs).sum(dim=1).mean()


class TV(BaseDivergence):
    """Total Variation Distance"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Use torch.where instead of creating a factor tensor
        diff = torch.abs(logits - targets)
        return diff.sum(dim=1).mean() * 0.5


class X2(BaseDivergence):
    """Chi-Square Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Simplified computation
        diff_squared = torch.pow(targets - logits, 2)
        loss = diff_squared / (logits + EPS_MEDIUM)
        return loss.sum(dim=1).mean() * 0.5


class PowD(BaseDivergence):
    """Power Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] <= 0:
            raise ValueError("Invalid alpha parameter for Power Divergence")

        logits, targets = self.prepare_inputs(logits, targets)
        alpha = self.param[1]

        C = torch.pow(logits, alpha - 1)
        loss = (torch.pow(targets, alpha) - C * logits) / (C + EPS_MEDIUM)
        return loss.sum(dim=1).mean()


class JS(BaseDivergence):
    """Jensen-Shannon Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # More numerically stable computation
        mixture = 0.5 * (logits + targets)
        log_mixture = torch.log(torch.clamp(mixture, min=EPS_TINY))
        log_logits = torch.log(torch.clamp(logits, min=EPS_TINY))
        log_targets = torch.log(torch.clamp(targets, min=EPS_TINY))

        loss = 0.5 * (
                targets * (log_targets - log_mixture) +
                logits * (log_logits - log_mixture)
        )
        return loss.sum(dim=1).mean()


class GenKL(BaseDivergence):
    """Generalized Kullback-Leibler Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] <= 0:
            raise ValueError("Invalid alpha parameter for Generalized KL")

        logits, targets = self.prepare_inputs(logits, targets)
        alpha = self.param[1]

        C = torch.pow(logits, alpha - 1)
        loss = ((C * logits) - torch.pow(targets, alpha)) / (alpha * (C + EPS_SMALL))
        return loss.sum(dim=1).mean()


class Exp(BaseDivergence):
    """Exponential Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] <= 0:
            raise ValueError("Invalid parameter for Exponential Divergence")

        logits, targets = self.prepare_inputs(logits, targets)
        # More stable computation with torch.clamp
        denom = torch.clamp(logits + EPS_LARGE * 10, min=EPS_TINY)
        exp_term = torch.exp((targets - logits) / denom) - 1
        loss = logits * exp_term
        return loss.sum(dim=1).mean()


class LeCam(BaseDivergence):
    """Le Cam Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] <= 0:
            raise ValueError("Invalid parameter for Le Cam Divergence")

        logits, targets = self.prepare_inputs(logits, targets)
        diff = logits - targets
        denom = logits + targets + EPS_MEDIUM
        loss = 0.5 * (diff * logits / denom)
        return loss.sum(dim=1).mean()


class AlphaRenyi(BaseDivergence):
    """Alpha-Renyi Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] < 0:
            raise ValueError("Invalid alpha parameter for Alpha-Renyi Divergence")

        logits, targets = self.prepare_inputs(logits, targets)
        alpha = self.param[1]

        # Handle alpha == 1 specially (limit case is KL divergence)
        if abs(alpha - 1.0) < EPS_TINY:
            # Return KL divergence for alpha ≈ 1
            log_ratio = torch.log(torch.clamp(targets / torch.clamp(logits, min=EPS_TINY), min=EPS_TINY))
            return (targets * log_ratio).sum(dim=1).mean()

        P = torch.pow(targets, alpha)
        Q = torch.pow(logits, 1 - alpha)
        loss = P * Q
        loss = torch.log(torch.clamp(loss.sum(dim=1), min=EPS_MEDIUM))
        return loss.mean() / (alpha - 1)


class BetaSkew(BaseDivergence):
    """Beta-Skew Divergence"""

    def __init__(self, nclass: int, param: Optional[List[float]] = None,
                 softmax_logits: bool = True, softmax_gt: bool = True):
        super(BetaSkew, self).__init__(nclass, param, softmax_logits, softmax_gt)
        if len(self.param) <= 1 or not 0 < self.param[1] < 1:
            raise ValueError("Invalid beta parameter for Beta-Skew Divergence")
        self.beta = self.param[1]
        # Create KL instance only once during initialization
        self.kl = KL(nclass, param, softmax_logits=False, softmax_gt=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Use direct tensor operation for mixing
        mixed = logits * (1 - self.beta) + targets * self.beta
        return self.kl(mixed, targets)


class CauchySchwarz(BaseDivergence):
    """Cauchy-Schwarz Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)

        # Compute dot product of logits and targets - batchwise
        inner_prod = (logits * targets).sum(dim=1)

        # Compute squared L2 norms - batchwise
        logits_norm_sq = (logits * logits).sum(dim=1)
        targets_norm_sq = (targets * targets).sum(dim=1)

        # Compute the loss with clamping for numerical stability
        norm_product = torch.clamp(logits_norm_sq * targets_norm_sq, min=EPS_MEDIUM)
        cs_ratio = torch.clamp(inner_prod * inner_prod / norm_product, min=EPS_TINY)
        loss = -0.5 * torch.log(cs_ratio)
        return loss.mean()


class Hellinger(BaseDivergence):
    """Hellinger Distance"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Compute square root with numerical stability
        sqrt_logits = torch.sqrt(torch.clamp(logits, min=EPS_TINY))
        sqrt_targets = torch.sqrt(torch.clamp(targets, min=EPS_TINY))
        # Compute Hellinger distance
        sqrt_diff = sqrt_logits - sqrt_targets
        return torch.norm(sqrt_diff, dim=1, p=2).mean() / math.sqrt(2)


class Bhattacharyya(BaseDivergence):
    """Bhattacharyya Distance"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Compute Bhattacharyya coefficient (BC)
        bc = torch.sqrt(torch.clamp(logits * targets, min=EPS_TINY)).sum(dim=1)
        # Bhattacharyya distance = -ln(BC)
        return -torch.log(torch.clamp(bc, min=EPS_TINY)).mean()


class Jeffrey(BaseDivergence):
    """Jeffrey's Divergence (symmetrized KL)"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Compute logs with numerical stability
        log_logits = torch.log(torch.clamp(logits, min=EPS_TINY))
        log_targets = torch.log(torch.clamp(targets, min=EPS_TINY))
        # KL(p||q)
        kl_pt = (targets * (log_targets - log_logits)).sum(dim=1)
        # KL(q||p)
        kl_qp = (logits * (log_logits - log_targets)).sum(dim=1)
        # Jeffrey = 0.5 * (KL(p||q) + KL(q||p))
        return 0.5 * (kl_pt + kl_qp).mean()


class ItakuraSaito(BaseDivergence):
    """Itakura-Saito Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Compute ratio with numerical stability
        safe_logits = torch.clamp(logits, min=EPS_TINY)
        ratio = torch.clamp(targets / safe_logits, min=EPS_TINY)
        # IS(p||q) = p/q - log(p/q) - 1
        return (ratio - torch.log(ratio) - 1).sum(dim=1).mean()


class SquaredHellinger(BaseDivergence):
    """Squared Hellinger Distance"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Compute square roots with numerical stability
        sqrt_logits = torch.sqrt(torch.clamp(logits, min=EPS_TINY))
        sqrt_targets = torch.sqrt(torch.clamp(targets, min=EPS_TINY))
        # Squared Hellinger = 0.5 * sum((sqrt(p) - sqrt(q))²)
        sqrt_diff = sqrt_logits - sqrt_targets
        return 0.5 * (sqrt_diff * sqrt_diff).sum(dim=1).mean()


class Tsallis(BaseDivergence):
    """Tsallis Divergence"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] <= 0:
            raise ValueError("Invalid alpha parameter for Tsallis Divergence")

        logits, targets = self.prepare_inputs(logits, targets)
        alpha = self.param[1]

        # Handle alpha ≈ 1 (KL divergence)
        if abs(alpha - 1.0) < EPS_TINY:
            safe_logits = torch.clamp(logits, min=EPS_TINY)
            log_ratio = torch.log(torch.clamp(targets / safe_logits, min=EPS_TINY))
            return (targets * log_ratio).sum(dim=1).mean()

        # Compute Tsallis divergence for alpha ≠ 1
        # D_α(p||q) = (1 - ∑ p^α * q^(1-α)) / (α - 1)
        term = (targets.pow(alpha) * logits.pow(1 - alpha)).sum(dim=1)
        return ((1 - term) / (alpha - 1)).mean()


class fDivergence(BaseDivergence):
    """
    General f-Divergence with customizable f function

    The f-divergence is defined as:
    D_f(P||Q) = ∑ q(x) * f(p(x)/q(x))

    where f is a convex function satisfying f(1) = 0
    """

    def __init__(self, nclass: int, param: Optional[List[float]] = None,
                 softmax_logits: bool = True, softmax_gt: bool = True,
                 f_func: Optional[Callable] = None):
        super(fDivergence, self).__init__(nclass, param, softmax_logits, softmax_gt)
        # Default to KL-divergence if no function provided
        self.f_func = f_func if f_func is not None else lambda x: x * torch.log(x) - x + 1

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Compute ratio with numerical stability
        safe_logits = torch.clamp(logits, min=EPS_TINY)
        ratio = torch.clamp(targets / safe_logits, min=EPS_TINY)
        # Compute f-divergence: D_f(P||Q) = ∑ q(x) * f(p(x)/q(x))
        f_values = self.f_func(ratio)
        return (logits * f_values).sum(dim=1).mean()


class WassersteinApprox(BaseDivergence):
    """
    Approximation of Wasserstein Distance for probability vectors

    Note: This is a simplified approximation based on cumulative distribution functions.
    For exact Wasserstein distance, optimal transport computation is required.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Compute absolute difference of cumulative sums (CDFs)
        logits_cdf = torch.cumsum(logits, dim=1)
        targets_cdf = torch.cumsum(targets, dim=1)
        return torch.abs(logits_cdf - targets_cdf).sum(dim=1).mean()


class RenyiDivergence(BaseDivergence):
    """
    Rényi Divergence

    Different from Alpha-Renyi, this implementation follows the more
    standard definition of Rényi divergence with a different normalization.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(self.param) <= 1 or self.param[1] <= 0:
            raise ValueError("Invalid alpha parameter for Rényi Divergence")

        logits, targets = self.prepare_inputs(logits, targets)
        alpha = self.param[1]

        # Handle special cases
        if abs(alpha - 1.0) < EPS_TINY:  # α ≈ 1 (KL divergence)
            safe_logits = torch.clamp(logits, min=EPS_TINY)
            log_ratio = torch.log(torch.clamp(targets / safe_logits, min=EPS_TINY))
            return (targets * log_ratio).sum(dim=1).mean()

        if alpha < EPS_TINY:  # α ≈ 0 (limit case)
            return torch.log(torch.clamp(logits / torch.clamp(targets, min=EPS_TINY),
                                         min=EPS_TINY)).max(dim=1)[0].mean()

        if alpha > 1e6:  # α ≈ ∞ (limit case)
            return torch.log(torch.clamp(targets / torch.clamp(logits, min=EPS_TINY),
                                         min=EPS_TINY)).max(dim=1)[0].mean()

        # Standard Rényi divergence for 0 < α < ∞, α ≠ 1
        ratio = torch.clamp(targets / torch.clamp(logits, min=EPS_TINY), min=EPS_TINY)
        power_ratio = torch.pow(ratio, alpha)
        avg_term = (logits * power_ratio).sum(dim=1)
        safe_avg = torch.clamp(avg_term, min=EPS_TINY)
        return (1.0 / (alpha - 1.0)) * torch.log(safe_avg).mean()


class Triangular(BaseDivergence):
    """
    Triangular Discrimination

    A bounded f-divergence: 0 ≤ Δ(P||Q) ≤ 2
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Compute squared difference
        diff = logits - targets
        diff_squared = diff * diff
        # Denominator with numerical stability
        denom = torch.clamp(logits + targets, min=EPS_TINY)
        # Triangular discrimination: Δ(P||Q) = ∑ (p - q)²/(p + q)
        return (diff_squared / denom).sum(dim=1).mean()


class LogitBCE(BaseDivergence):
    """
    Logit-based Binary Cross-Entropy

    A variant that works with logits directly, similar to BCEWithLogitsLoss.
    """

    def __init__(self, nclass: int, param: Optional[List[float]] = None,
                 softmax_logits: bool = False, softmax_gt: bool = True):
        # Note: we set softmax_logits=False by default
        super(LogitBCE, self).__init__(nclass, param, softmax_logits, softmax_gt)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # For this loss, we handle raw logits
        _, targets = self.prepare_inputs(logits, targets)

        # Use PyTorch's native implementation for stability
        return F.binary_cross_entropy_with_logits(logits, targets)


class BrierScore(BaseDivergence):
    """Brier Score - measures mean squared error between probability distributions"""
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        return torch.mean((logits - targets) ** 2)


class FocalLoss(BaseDivergence):
    """Focal Loss - focuses on hard-to-classify examples"""
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gamma = self.param[0] if len(self.param) > 0 else 2.0
        alpha = self.param[1] if len(self.param) > 1 else 0.25

        logits, targets = self.prepare_inputs(logits, targets)
        logits = torch.clamp(logits, min=EPS_TINY, max=1.0)
        BCE = -targets * torch.log(logits)
        focal = alpha * ((1 - logits) ** gamma) * BCE
        return focal.sum(dim=1).mean()


class EnergyDistance(BaseDivergence):
    """Energy Distance - compares distributions using pairwise Euclidean norms"""
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        ed = torch.norm(logits - targets, p=2, dim=1).mean()
        self_ed1 = torch.norm(logits.unsqueeze(1) - logits.unsqueeze(0), dim=2).mean()
        self_ed2 = torch.norm(targets.unsqueeze(1) - targets.unsqueeze(0), dim=2).mean()
        return 2 * ed - self_ed1 - self_ed2


class MMD(BaseDivergence):
    """Maximum Mean Discrepancy with RBF kernel"""
    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        y_sq = (y ** 2).sum(dim=1, keepdim=True)
        xy = x @ y.T
        dist = x_sq + y_sq.T - 2 * xy
        return torch.exp(-dist / (2 * sigma ** 2))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sigma = self.param[0] if len(self.param) > 0 else 1.0
        logits, targets = self.prepare_inputs(logits, targets)

        K_XX = self.rbf_kernel(logits, logits, sigma)
        K_YY = self.rbf_kernel(targets, targets, sigma)
        K_XY = self.rbf_kernel(logits, targets, sigma)

        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()


class SinkhornDistance(BaseDivergence):
    """Sinkhorn Approximation of Wasserstein Distance using CDF difference"""
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        cdf_logits = torch.cumsum(logits, dim=1)
        cdf_targets = torch.cumsum(targets, dim=1)
        return torch.abs(cdf_logits - cdf_targets).sum(dim=1).mean()
# Dictionary-based factory defined at module level for efficiency
_LOSS_CLASSES = {
    "KL": KL,
    "TV": TV,
    "X2": X2,
    "PowD": PowD,
    "JS": JS,
    "GenKL": GenKL,
    "Exp": Exp,
    "LeCam": LeCam,
    "AlphaRenyi": AlphaRenyi,
    "BetaSkew": BetaSkew,
    "CauchySchwarz": CauchySchwarz,
    "Hellinger": Hellinger,
    "Bhattacharyya": Bhattacharyya,
    "Jeffrey": Jeffrey,
    "ItakuraSaito": ItakuraSaito,
    "SquaredHellinger": SquaredHellinger,
    "Tsallis": Tsallis,
    "fDivergence": fDivergence,
    "WassersteinApprox": WassersteinApprox,
    "RenyiDivergence": RenyiDivergence,
    "Triangular": Triangular,
    "LogitBCE": LogitBCE,
"FocalLoss": FocalLoss,
"BrierScore": BrierScore,
"EnergyDistance": EnergyDistance,
"MMD": MMD,
"SinkhornDistance": SinkhornDistance,
}




def get_loss(name: str, param: Optional[List[float]] = None, nclass: int = 10,
             softmax_logits: bool = True, softmax_gt: bool = True) -> BaseDivergence:
    """
    Factory function to create loss objects.

    Args:
        name: Name of the divergence measure
        param: Parameters for the divergence measure
        nclass: Number of classes
        softmax_logits: Whether to apply softmax to model predictions
        softmax_gt: Whether to apply softmax to ground truth distributions

    Returns:
        Instance of the specified divergence measure

    Raises:
        ValueError: If an invalid loss function name is provided
    """
    if name in _LOSS_CLASSES:
        return _LOSS_CLASSES[name](nclass, param, softmax_logits, softmax_gt)

    raise ValueError(f"Invalid loss function: {name}. Available options: {', '.join(_LOSS_CLASSES.keys())}")