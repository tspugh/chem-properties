from dataclasses import dataclass
from typing import List, Union, Tuple, Optional, Dict, TypedDict, Any
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    MessagePassing, 
    GATConv, 
    JumpingKnowledge,
    InnerProductDecoder,
    global_mean_pool,
    VGAE,
    GCNConv,
    LayerNorm,
)
from torch_geometric.utils import (
    add_self_loops, 
    dropout_edge,
    dropout_adj,
    dropout_node,
    negative_sampling,
    to_undirected
)
from torch_geometric.nn.resolver import (
    normalization_resolver,
    activation_resolver
)
from torch_geometric.data import Data, Batch

from src.aux_graph_models import KMeansCarry, KMeansHeadConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphHRMLayerConfig(TypedDict):

    cycles: int
    layers: int
    hidden_dim: int

    dropout: float

class HRM_Carry(ABC):
    @abstractmethod
    @property
    def z_H(self) -> Any:
        pass

    @abstractmethod
    @property
    def z_L(self) -> Any:
        pass

class KMeansGraphHRMCarry(HRM_Carry):

    def __init__(self, z_H_nodes: Tensor, z_H_edges: Tensor, z_L: KMeansCarry):
        self.z_H_nodes = z_H_nodes
        self.z_H_edges = z_H_edges
        self.z_L = z_L

    @property
    def z_H(self) -> Tuple[Tensor, Tensor]:
        return self.z_H_nodes, self.z_H_edges

    @property
    def z_L(self) -> KMeansCarry:
        return self.z_L

@dataclass
class HRM_TotalCarry:

    inner_carry: HRM_Carry

    steps: Tensor
    halted: Tensor

    current_data: Dict[str, Tensor]



class GraphKMeansHRMKMeansModuleConfig(TypedDict):
    batch_size: int
    node_count: int
    
    kmeans_head_config: KMeansHeadConfig
    kmeans_module_block_config: GraphHRMKMeansModuleBlockConfig

    H_cycles: int
    K_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"