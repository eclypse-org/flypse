"""Data abstractions for federated learning."""

from flypse.data.dataset import (
    DataSource,
    FederatedDataset,
)
from flypse.data.dirichlet import DirichletDataset
from flypse.data.uniform import UniformDataset
from flypse.data.view import DataView

__all__ = [
    "DataSource",
    "DataView",
    "DirichletDataset",
    "FederatedDataset",
    "UniformDataset",
]
