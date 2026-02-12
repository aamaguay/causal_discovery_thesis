
from torch.nn import functional as F
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform

from causa.utils import build_coupling_flow

#######################################################################################################
#### Simple NSF PiecewiseRationalQuadraticCouplingTransform ###########################################

class SimpleNSF(Flow):
    """
    Simplified Neural Spline Flow using piecewise rational quadratic
    coupling transforms with checkerboard masking.
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        num_bins=10,
        tails="linear",
        tail_bound=3.0,
        dropout_probability=0.0,
        activation=F.relu,
        apply_unconditional_transform=False,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):
        def coupling_factory(mask, create_net):
            return PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_net,
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                apply_unconditional_transform=apply_unconditional_transform,
            )

        transform = build_coupling_flow(
            features=features,
            num_layers=num_layers,
            coupling_factory=coupling_factory,
            hidden_features=hidden_features,
            num_blocks_per_layer=num_blocks_per_layer,
            activation=activation,
            dropout_probability=dropout_probability,
            batch_norm_within_layers=batch_norm_within_layers,
            batch_norm_between_layers=batch_norm_between_layers,
        )

        super().__init__(
            transform=transform,
            distribution=StandardNormal([features]),
        )



