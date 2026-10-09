"""Model compression: low-rank MLP approximation, neuron pruning."""

from .lowrank_mlp import setup_lowrank_mlp  # noqa: F401
from .neuron_prune import setup_mlp_neuron_pruning  # noqa: F401
