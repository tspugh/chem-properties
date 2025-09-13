from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union
import inspect
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    ChebConv,
    GATConv,
    GCNConv,
    GlobalAttention,
    InnerProductDecoder,
    LayerNorm,
    NNConv,
    MessagePassing,
    VGAE,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    Linear
)
from torch_geometric.nn.pool import radius
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.utils import (
    add_self_loops,
    bipartite_subgraph,
    dropout_adj,
    dropout_edge,
    dropout_node,
    negative_sampling,
    subgraph,
    to_undirected,
)

from torch_scatter import scatter_min

# HEAVILY REFERENCES https://github.com/sapientinc/HRM/blob/05dd4ef795a98c20110e380a330d0b3ec159a46b/models/hrm/hrm_act_v1.py#L222

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KMeansCarry(Data):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = kwargs.get('mask', None)
        self.none_selected = kwargs.get('none_selected', None)
    
    @property
    def k(self) -> int:
        if self.mask is not None:
            return self.mask.shape[-1]  # Last dimension is k
        return 0
    
    def masked_x(self, mask_idx: Optional[int] = None) -> Tensor:
        """Get masked node features for a specific mask index"""
        if self.mask is None or mask_idx is not None and mask_idx >= self.mask.shape[-1]:
            return self.x
        if mask_idx is None:
            return torch.stack([self.x * self.mask[:, i:i+1] for i in range(self.mask.shape[-1])], dim=0)
        return self.x * self.mask[:, mask_idx:mask_idx+1]
    
    def masked_edge_index(self, mask_idx: int) -> Tensor:
        """Get masked edge index for a specific mask index"""
        if self.mask is None or mask_idx >= self.mask.shape[-1]:
            return self.edge_index
        node_mask = self.mask[:, mask_idx] > 0
        if node_mask.sum() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.edge_index.device)
        return subgraph(node_mask, self.edge_index)[0]

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'mask' or key == 'none_selected':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

# Takes in a mask and feature nodes, outputs a new mask
class KMeansHeadConfig(TypedDict):
    node_count: int
    node_dim: int
    max_nodes: int

    num_layers: int

    # In: mask, features; Out: weighted_features
    weighting_module: nn.Module
    # In: mask, weighted_features; Out: center_mask
    center_module: nn.Module
    # In: center_mask, weighted_features; Out: mask
    mask_module: nn.Module

    act: str
    act_kwargs: Dict[str, Any]
    dropout: float


class KMeansHead(nn.Module):

    def _validate_module_shapes(self, module: nn.Module, expected_in: int, expected_out: int, module_name: str):
        """Validate input and output shapes of a module"""
        if hasattr(module, 'in_channels') and module.in_channels != expected_in:
            raise ValueError(f"{module_name} input channels mismatch. "
                           f"Expected {expected_in}, got {module.in_channels}")
        
        if hasattr(module, 'out_channels') and module.out_channels != expected_out:
            raise ValueError(f"{module_name} output channels mismatch. "
                           f"Expected {expected_out}, got {module.out_channels}")

    def __init__(self, config: KMeansHeadConfig, training: bool = True):
        super(KMeansHead, self).__init__()
        self.config = config
        self.in_channels = config["node_dim"]
        self.training = training
        
        self.weighting_module = config["weighting_module"]
        self.center_module = config["center_module"]
        self.mask_module = config["mask_module"]

        # Validate module shapes
        self._validate_module_shapes(
            self.weighting_module, 
            config["node_dim"], 
            config["node_dim"], 
            "Weighting module"
        )
        self._validate_module_shapes(
            self.center_module, 
            config["node_dim"], 
            config["node_count"], 
            "Center module"
        )
        self._validate_module_shapes(
            self.mask_module, 
            config["node_dim"], 
            config["node_count"], 
            "Mask module"
        )
        self.act = activation_resolver(config["act"], **config["act_kwargs"])
        self.dropout = config["dropout"]


    def forward(self, k_data: KMeansCarry, mask_idx: int, batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for a single sample (no batching).
        
        Args:
            k_data: KMeansCarry object containing nodes, mask, none_selected, and edge_index
            
        Returns:
            KMeansCarry object with updated mask
        """
        # Process through the three modules
        mask_weighting = self.weighting_module(k_data, mask_idx, batch)
        center_nodes, center_batch = self.center_module(k_data, mask_idx, mask_weighting, batch)
        output_mask = self.mask_module(k_data, mask_idx, center_nodes, batch, center_batch)

        if not output_mask.dtype == torch.bool:
            output_mask=torch.gt(self.act(output_mask).squeeze(-1), 0).bool()

        return output_mask
    
    def reset_parameters(self):
        """Reset parameters for all modules in KMeansHead"""
        if hasattr(self.weighting_module, 'reset_parameters'):
            self.weighting_module.reset_parameters()
        if hasattr(self.center_module, 'reset_parameters'):
            self.center_module.reset_parameters()
        if hasattr(self.mask_module, 'reset_parameters'):
            self.mask_module.reset_parameters()


class SpectralWeightingConfig(TypedDict):
    node_channels: int
    K: int
    num_layers: int
    normalization: Optional[str]  # 'sym', 'rw', or None
    bias: bool
    dropout: float
    norm: str
    norm_kwargs: Dict[str, Any]


class DiscreteMeanCenterConfig(TypedDict):
    distance_metric: Literal['euclidean', 'cosine', 'manhattan']

class SpectralWeighting(nn.Module):
    """
    Spectral weighting module based on Chebyshev spectral graph convolution.
    Uses a mask to minimize calculations by only including nodes where mask == 1.
    Supports multiple layers for deeper spectral processing.
    """
    
    def __init__(self, config: SpectralWeightingConfig, training: bool = True):
        super(SpectralWeighting, self).__init__()
        self.config = config
        self.num_layers = config["num_layers"]
        self.training = training
        # Create multiple ChebConv layers
        self.cheb_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = config["dropout"]
        
        # First layer: in_channels -> out_channels
        self.cheb_convs.append(ChebConv(
            in_channels=config["node_channels"],
            out_channels=config["node_channels"],
            K=config["K"],
            normalization=config["normalization"],
            bias=config["bias"]
        ))
        
        # Additional layers: out_channels -> out_channels
        for _ in range(self.num_layers - 1):
            self.cheb_convs.append(ChebConv(
                in_channels=config["node_channels"],
                out_channels=config["node_channels"],
                K=config["K"],
                normalization=config["normalization"],
                bias=config["bias"]
            ))
        
        # Create normalization and dropout layers for each layer
        for _ in range(self.num_layers):
            self.norms.append(normalization_resolver(config["norm"], **config["norm_kwargs"]))
    
    def forward(self, k_data: KMeansCarry, mask_idx: int, batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through multiple spectral layers.
        
        Args:
            masked_features: Node features with mask applied
            edge_index: Edge connectivity tensor
            batch: Batch assignment tensor for batched processing
            
        Returns:
            Weighted node features after processing through all layers
        """
        x = k_data.masked_x(mask_idx)
        # Process through each layer
        for i in range(self.num_layers):
            # Apply Chebyshev spectral convolution
            x = self.cheb_convs[i](
                x=x,
                edge_index=k_data.masked_edge_index(mask_idx)
            )
            
            if self.norms[i] is not None:
                x = self.norms[i](x)
        
        return x
    
    def reset_parameters(self):
        """Reset parameters for all ChebConv layers, norms, and dropouts"""
        for conv in self.cheb_convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()


class EuclideanDistance(nn.Module):
    """Euclidean distance module: sqrt(sum((x - c)^2))"""
    
    def forward(self, features: Tensor, center: Tensor) -> Tensor:
        return torch.norm(features - center.unsqueeze(0), dim=1)
    
    def reset_parameters(self):
        """EuclideanDistance has no learnable parameters"""
        pass


class CosineDistance(nn.Module):
    """Cosine distance module: 1 - cosine_similarity"""
    
    def forward(self, features: Tensor, center: Tensor) -> Tensor:
        # Normalize features and center
        features_norm = F.normalize(features, p=2, dim=1)
        center_norm = F.normalize(center.unsqueeze(0), p=2, dim=1)
        cosine_sim = torch.sum(features_norm * center_norm, dim=1)
        return 1 - cosine_sim


class ManhattanDistance(nn.Module):
    """Manhattan distance module: sum(|x - c|)"""
    
    def forward(self, features: Tensor, center: Tensor) -> Tensor:
        return torch.sum(torch.abs(features - center.unsqueeze(0)), dim=1)


class DiscreteMeanCenter(nn.Module):
    """
    Discrete mean center module that calculates the weighted mean of features
    and returns a mask with the closest node to the mean set to 1.
    """
    
    def __init__(self, config: DiscreteMeanCenterConfig, training: bool = True):
        super(DiscreteMeanCenter, self).__init__()
        self.config = config
        self.training = training

        # Construct distance module based on metric
        distance_metric = config["distance_metric"]
        if distance_metric == 'euclidean':
            self.distance_module = nn.PairwiseDistance(p=2)
        elif distance_metric == 'cosine':
            self.distance_module = nn.CosineSimilarity(dim=1)
        elif distance_metric == 'manhattan':
            self.distance_module = nn.PairwiseDistance(p=1)
        else:
            raise ValueError(f"Invalid distance metric: {distance_metric}. "
                           f"Must be one of ['euclidean', 'cosine', 'manhattan']")
    
    def forward(self, k_data: KMeansCarry, mask_idx: int, weighted_features: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Calculate weighted mean and return mask for closest node.
        
        Args:
            k_data: KMeansCarry object containing clustering state
            mask_idx: Index of the current mask being processed
            weighted_features: Weighted node features tensor
            batch: Batch assignment tensor for batched processing
            
        Returns:
            New mask tensor with closest node to mean set to 1, rest to 0
        """
        if batch is not None and hasattr(k_data, 'num_graphs'):
            distances = torch.zeros(k_data.x.shape[0], dtype=torch.float, device=k_data.x.device)
            for i in range(k_data.num_graphs):
                data = k_data.get_example(i)
                graph_weighted_features = weighted_features[data.mask[mask_idx]]
                distances += self.distance(graph_weighted_features)
        else:
            distances = self.distance(weighted_features)
        
        # Find the node with minimum distance
        if batch is not None:
            _, closest_idxs = scatter_min(torch.pow(distances, 2), batch)
        else:
            _, closest_idxs = scatter_min(torch.pow(distances, 2))

        center_batch = None
        if batch is not None:
            center_batch = batch[closest_idxs]
        
        return k_data.x[closest_idxs], center_batch
    
    def distance(self, weighted_features: Tensor) -> Tensor:
        """Process a single graph's features."""
        # Calculate weighted mean (sum of weighted features / number of features)
        epsilon = 1e-8
        num_features = weighted_features.shape[0]
        center = torch.sum(weighted_features, dim=0) / (num_features + epsilon)
        
        # Calculate distances from all nodes to the center
        if self.config["distance_metric"] == 'cosine':
            # For cosine distance, we need different handling
            features_norm = F.normalize(weighted_features, p=2, dim=1)
            center_norm = F.normalize(center.unsqueeze(0), p=2, dim=1)
            cosine_sim = torch.sum(features_norm * center_norm, dim=1)
            distances = 1 - cosine_sim
        else:
            # For euclidean and manhattan
            distances = self.distance_module(weighted_features, center.unsqueeze(0).expand_as(weighted_features))
        
        return distances
    


class RadiusMaskConfig(TypedDict):
    max_num_neighbors: int
    radius: int
    weighting_module: nn.Module
    threshold: Optional[float]
    node_dim: int


class RadiusAttentionWeights(nn.Module):
    """
    Radius-based mask module that finds nodes within a specified radius from center nodes.
    Uses torch_geometric.nn.pool.radius to efficiently find neighbors.
    """
    
    def __init__(self, config: RadiusMaskConfig):
        """
        Initialize RadiusAttentionWeights module.

        Args:
            config: Configuration dictionary containing:
                max_num_neighbors: Maximum number of nodes to include in the mask
                radius: Radius within which to search for nodes
                weighting_module: Module to weight the features
        """
        super(RadiusAttentionWeights, self).__init__()
        self.max_num_neighbors = config["max_num_neighbors"]
        self.radius = config["radius"]
        self.threshold = config.get("threshold", 0.1)

        self.weighting_module = config["weighting_module"]

        self._mask_linear = torch.nn.Linear(config["node_dim"], 1)
    
    def forward(self, k_data: KMeansCarry, mask_idx: int, center: Tensor, batch: Optional[Tensor] = None, center_batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for a single sample (no batching).
        
        Args:
            center_mask: Binary mask tensor indicating center nodes (1 for center, 0 otherwise)
            features: Node features tensor
            edge_index: Edge connectivity tensor
            
        Returns:
            New mask tensor with nodes within radius set to 1, rest to 0
        """

        if center.shape[0] == 0:
            # No center nodes, return empty mask
            return k_data.mask[mask_idx]
        
        # Use radius function to find neighbors
        # x: all node features, y: center node features
        row, col = radius(x=k_data.x, y=center, r=self.radius, batch_x=batch, batch_y=center_batch, max_num_neighbors=self.max_num_neighbors)
        
        # Create new mask
        new_mask_idx_pre = torch.unique(torch.stack([row, col], dim=0))
        arange_idx = torch.arange(k_data.x.shape[0], device=k_data.x.device)
        new_mask_idx = torch.isin(arange_idx, new_mask_idx_pre)
        
        k_data_c_edge_index, k_data_c_edge_attr = subgraph(col, k_data.edge_index, k_data.edge_attr)
        
        # Apply weighting module if available       
        sig = inspect.signature(self.weighting_module.forward)

        if 'batch' in sig.parameters:
            weighting = self.weighting_module(k_data.x, edge_index=k_data_c_edge_index, edge_attr=k_data_c_edge_attr, batch=batch)
        else:
            weighting = self.weighting_module(k_data.x, edge_index=k_data_c_edge_index, edge_attr=k_data_c_edge_attr) 

        w = self._mask_linear(weighting)
        w = F.relu(w)
        w = w.squeeze(-1)

        assert w.shape == new_mask_idx.shape, f"Weighting and new mask index shape mismatch, got {w.shape} and {new_mask_idx.shape}"

        new_mask_idx = torch.logical_and(new_mask_idx, torch.gt(w, self.threshold))
        
        return new_mask_idx
    
    def reset_parameters(self):
        """Reset parameters for weighting module"""
        if hasattr(self.weighting_module, 'reset_parameters'):
            self.weighting_module.reset_parameters()
        self._mask_linear.reset_parameters()

class KMeansConfig(TypedDict):
    k: int
    max_iter: int
    thresh: float = 1e-8
    max_overlap: int = 2
    head_module: KMeansHeadConfig
    excluded_is_cluster: bool


class KMeans(nn.Module):
    """
    PyTorch Geometric KMeans module with k heads.
    - Initializes k `KMeansHead` modules from the provided `KMeansConfig`.
    - Forward takes a `Data` object and a tensor of k masks and returns a new tensor of masks.
    - Optionally applies a graph attention convolution over the head weights before assignment.
    """
    def __init__(self, config: KMeansConfig, training: bool = True):
        super(KMeans, self).__init__()
        self.config = config
        self.k = config["k"]
        self.max_iter = config["max_iter"]
        self.thresh = config.get("thresh", 1e-8)
        self.max_overlap = min(config.get("max_overlap", self.k//2), self.k)
        self.excluded_is_cluster = config["excluded_is_cluster"]

        if self.max_overlap <= 0:
            raise ValueError(f"max_overlap must be greater than 0, got {self.max_overlap}")
        if self.k <= 0:
            raise ValueError(f"k must be greater than 0, got {self.k}")

        head_config = config["head_module"]
        self.heads = nn.ModuleList([
            KMeansHead(head_config, training=training) for _ in range(self.k)
        ])
    
    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()

    def _normalize_mask_input(self, masks: Tensor, num_nodes: int) -> Tensor:
        """
        Ensure mask tensor has shape [num_nodes, k]. If provided as [k, num_nodes],
        transpose it. Raise if shape is invalid.
        """
        if masks.dim() == 1 and masks.shape[0] == num_nodes:
            return masks.unsqueeze(1)
        if masks.dim() != 2:
            raise ValueError(f"Expected 2D mask tensor, got shape {tuple(masks.shape)}")
        dim0, dim1 = masks.shape
        if dim0 == num_nodes and dim1 == self.k:
            return masks
        if dim0 == self.k and dim1 == num_nodes:
            return masks.t()
        raise ValueError(f"Expected masks with shape [num_nodes, k] or [k, num_nodes], got {tuple(masks.shape)}")

    def forward(self, kmeans_carry: KMeansCarry, batch: Optional[Tensor] = None) -> KMeansCarry:
        """
        Args:
            kmeans_carry: KMeansCarry object containing nodes, mask, none_selected, and edge_index
            batch: Batch assignment tensor for batched processing
        Returns:
            KMeansCarry object with updated mask
        """
        num_nodes = kmeans_carry.x.shape[0]
        
        # Create individual carries for each cluster/head
        head_results = [head(kmeans_carry, idx, batch=batch) for idx, head in enumerate(self.heads)]
        mask_tensor = torch.stack(head_results, dim=-1)  # [num_nodes, k]

        # Only consider weights strictly greater than threshold
        above_thresh = torch.greater(mask_tensor, self.thresh)
        masked_weights = kmeans_carry.x.norm(dim=1, keepdim=True).expand(-1, self.k).masked_fill(~above_thresh, float("-inf"))

        top_vals, top_idx = torch.topk(masked_weights, k=min(self.max_overlap, self.k), dim=1)
        valid = torch.isfinite(top_vals)

        # Create final mask tensor
        final_mask = torch.zeros_like(mask_tensor)
        row_idx = torch.arange(num_nodes, device=kmeans_carry.x.device).unsqueeze(1).expand_as(top_idx)
        row_idx_valid = row_idx[valid]
        col_idx_valid = top_idx[valid]
        final_mask[row_idx_valid, col_idx_valid] = 1

        # Optional extra cluster for nodes not assigned to any mask
        none_selected = torch.zeros(num_nodes, device=kmeans_carry.x.device, dtype=torch.bool)
        if self.excluded_is_cluster:
            none_selected = final_mask.sum(dim=1) == 0

        return KMeansCarry(
            x=kmeans_carry.x,
            edge_index=kmeans_carry.edge_index,
            batch=kmeans_carry.batch,
            mask=final_mask,
            none_selected=none_selected
        )





class VGAEEncoder(nn.Module):
    """VGAE Encoder for GraphHRMAttentionBlock"""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, 
                 layers: int, dropout: float, encoder_type: str = "gcn"):
        super(VGAEEncoder, self).__init__()
        self.encoder_type = encoder_type
        self.layers = layers
        
        # Build encoder layers
        if encoder_type == "gcn":
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, latent_dim))
        elif encoder_type == "gat":
            self.convs = nn.ModuleList()
            self.convs.append(GATConv(input_dim, hidden_dim // 8, heads=8, dropout=dropout))
            for _ in range(layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout))
            self.convs.append(GATConv(hidden_dim, latent_dim, heads=1, dropout=dropout))
        elif encoder_type == "cheb":
            self.convs = nn.ModuleList()
            self.convs.append(ChebConv(input_dim, hidden_dim, K=3))
            for _ in range(layers - 2):
                self.convs.append(ChebConv(hidden_dim, hidden_dim, K=3))
            self.convs.append(ChebConv(hidden_dim, latent_dim, K=3))
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass returning mu and logstd
        
        Args:
            x: Node features tensor
            edge_index: Edge connectivity tensor
            batch: Batch assignment tensor for batched processing
        """
        # PyTorch Geometric convolutions handle batching automatically via the data structure
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.training and self.dropout is not None:
                x = F.dropout(x, p=self.dropout)
        
        # Final layer for mean (mu)
        mu = self.convs[-1](x, edge_index)

        logstd = mu  # Simplified: using same output for logstd
        
        return mu, logstd

class OutputHeadConfig(TypedDict):
    node_dim: int
    hidden_dim: int
    output_dim: int
    pooling_type: str  # "mean", "max", "add", "attention"
    norm: str
    norm_kwargs: Dict[str, Any]
    act: str
    act_kwargs: Dict[str, Any]


class OutputHead(nn.Module):
    """
    Output head module that takes KMeansCarry nodes and edges as input
    and outputs a configurable length output vector.
    """
    
    def __init__(self, config: OutputHeadConfig, training: bool = True):
        super(OutputHead, self).__init__()
        self.config = config
        self.node_dim = config["node_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = config["output_dim"]
        self.pooling_type = config["pooling_type"]
        self.training = training
        
        # Graph pooling layer
        if self.pooling_type == "mean":
            self.pooling = global_mean_pool
        elif self.pooling_type == "max":
            self.pooling = global_max_pool
        elif self.pooling_type == "add":
            self.pooling = global_add_pool
        elif self.pooling_type == "attention":
            self.pooling = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(self.node_dim, 1),
                    nn.Sigmoid()
                )
            )
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        self.linear1 = Linear(self.node_dim, self.hidden_dim)
        self.linear2 = Linear(self.hidden_dim, self.output_dim)
        
        # Normalisierung und Aktivierung
        self.norm = normalization_resolver(config["norm"], **config["norm_kwargs"])
        # Make BatchNorm safe for batch size 1 during training
        try:
            from torch_geometric.nn.norm import BatchNorm as PyGBatchNorm  # type: ignore
        except Exception:
            PyGBatchNorm = None  # type: ignore
        if PyGBatchNorm is not None and isinstance(self.norm, PyGBatchNorm):
            if hasattr(self.norm, 'allow_single_element'):
                self.norm.allow_single_element = True  # type: ignore[attr-defined]
        # Fallback: if using torch.nn.BatchNorm1d, prefer LayerNorm to avoid N=1 error
        if isinstance(self.norm, nn.BatchNorm1d):
            self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = activation_resolver(config["act"], **config["act_kwargs"])
    
    def forward(self, kmeans_carry: KMeansCarry, batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for output head.
        
        Args:
            kmeans_carry: KMeansCarry object containing nodes and edge_index
            batch: Batch assignment tensor for batched processing
            
        Returns:
            Output tensor of shape [batch_size, output_dim] if batched, [output_dim] if single graph
        """
        nodes = kmeans_carry.x
        
        # Apply graph pooling
        # PyTorch Geometric pooling functions handle batch automatically

        
        pooled = self.pooling(nodes, batch=batch)
        
        #TODO - for each pool
        # Apply linear layers with normalization and activation
        x = self.linear1(pooled)
        if self.norm is not None:
            # Safe normalization for batch size 1 during training
            try:
                from torch_geometric.nn.norm import BatchNorm as PyGBatchNorm  # type: ignore
            except Exception:
                PyGBatchNorm = None  # type: ignore
            if isinstance(self.norm, nn.BatchNorm1d) or (PyGBatchNorm is not None and isinstance(self.norm, PyGBatchNorm)):
                if self.training and x.shape[0] <= 1:
                    # Fallback to LayerNorm behavior to avoid BN crash at N=1
                    x = F.layer_norm(x, (x.shape[-1],))
                else:
                    x = self.norm(x)
            else:
                x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        x = self.linear2(x)
        return x
    
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()
        if self.act is not None:
            self.act.reset_parameters()


class KMeansHRMInnerModuleConfig(TypedDict):
    add_self_loops: bool
    dropout: float
    hidden_dim: int
    edge_dim: int
    node_dim: int
    layers: int

    kmeans_config: KMeansConfig
    output_head_config: OutputHeadConfig
    policy_module_config: OutputHeadConfig
    add_negative_edges: bool
    K_cycles: int
    L_cycles: int
    batch_size: int
    halt_max_steps: int
    halt_exploration_prob: float


    vgae_encoder_type: str
    vgae_latent_dim: int
    vgae_encoder_layers: int
    vgae_encoder_dropout: float
    vgae_decoder_type: Optional[str]
    vgae_kl_weight: float

    



class KMeansHRMInnerModule(nn.Module):

    def __init__(self, config: KMeansHRMInnerModuleConfig, training: bool = True):
        super(KMeansHRMInnerModule, self).__init__()
        self.config = config
        self.add_self_loops = self.config['add_self_loops']
        self.dropout = self.config['dropout']
        self.hidden_dim = self.config['hidden_dim']
        self.node_dim = self.config['node_dim']
        self.layers = self.config['layers']
        self.edge_dim = self.config['edge_dim']

        self.latent_dim = self.config['vgae_latent_dim']

        self.kmeans_config = self.config['kmeans_config']
        self.k = self.kmeans_config['k']
        self.excluded_is_cluster = self.kmeans_config['excluded_is_cluster']

        self.output_head_config = self.config['output_head_config']
        self.policy_module_config = self.config['policy_module_config']
        self.policy_module_config['output_dim'] = 2 # 0 = continue, 1 = halt

        self.add_negative_edges = self.config['add_negative_edges']

        self.K_cycles = self.config['K_cycles']
        self.L_cycles = self.config['L_cycles']

        self.batch_size = self.config['batch_size']

        self.halt_max_steps = self.config['halt_max_steps']
        self.halt_exploration_prob = self.config['halt_exploration_prob']

        self.kmeans_module = KMeans(self.kmeans_config, training=training)

        prevgae_attention_layers = []
        prevgae_attention_layers.append(torch.nn.Linear(self.edge_dim, self.hidden_dim))
        prevgae_attention_layers.append(torch.nn.ReLU())
        for _ in range(max(0, self.layers-2)):
            prevgae_attention_layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
            prevgae_attention_layers.append(torch.nn.ReLU())
        prevgae_attention_layers.append(torch.nn.Linear(self.hidden_dim, self.node_dim * self.hidden_dim))
        self.prevgae_attention_layers = nn.Sequential(*prevgae_attention_layers)

        self.encode_conv = NNConv(
            in_channels=self.node_dim,
            out_channels=self.hidden_dim,
            nn=self.prevgae_attention_layers,
            aggr='mean'
        )

        self.linear_post_attention = nn.Sequential(
            nn.Linear(self.latent_dim, self.node_dim * 8),
            nn.ReLU(),
            (nn.Dropout(self.dropout if training else 0.0)),
            nn.Linear(self.node_dim * 8, self.node_dim),
            nn.LayerNorm(self.node_dim)
        )
        
        # Initialize VGAE if configured
        self.vgae_encoder = VGAEEncoder(
            input_dim=self.hidden_dim,  # Assuming node features dimension
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            layers=self.config['vgae_encoder_layers'],
            dropout=self.config['vgae_encoder_dropout'] if training else 0.0,
            encoder_type=self.config['vgae_encoder_type']
        )
        
        # Initialize VGAE model
        decoder = None
        if self.config.get('vgae_decoder_type') == "custom":
            # You can implement a custom decoder here
            decoder = InnerProductDecoder()  # Fallback to default
        
        self.vgae = VGAE(
            encoder=self.vgae_encoder,
            decoder=decoder
        )

        channels = self.k
        if self.excluded_is_cluster:
            channels += 1
        
        self.norm = LayerNorm(self.node_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

        self.output_head = OutputHead(self.output_head_config, training=training)
        self.policy_module = OutputHead(self.policy_module_config, training=training)

    def reset_parameters(self):
        """Reset parameters for all modules in 83Module"""
        self.kmeans_module.reset_parameters()
        self.vgae_encoder.reset_parameters()
        self.vgae.reset_parameters()
        if hasattr(self.prevgae_attention_layers, 'reset_parameters'):
            self.prevgae_attention_layers.reset_parameters()
        # Reset linear layers and norms safely
        for module in self.linear_post_attention:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        if hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()
        # Dropout has no parameters to reset
        self.output_head.reset_parameters()
        self.policy_module.reset_parameters()
        
        # Reset input embeddings if they exist
        if hasattr(self, 'input_embedding'):
            self.input_embedding.reset_parameters()
        if hasattr(self, 'puzzle_embedding'):
            self.puzzle_embedding.reset_parameters()
    
    
    def _link_subgraph_to_nodes(self, input_embeddings: Tensor, kmeans_nodes: Tensor, kmeans_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Link subgraph to nodes.
        
        Args:
            input_embeddings: Input embeddings tensor
            kmeans_nodes: KMeans nodes tensor
            kmeans_edge_index: KMeans edge index tensor
            
        Returns:
            Tuple of (bipartite_edge_index, kmean_offset_edges)
        """
        kmean_offset_edges = kmeans_edge_index.shape[1] + input_embeddings.shape[1]
        bipartite_edge_index = bipartite_subgraph(input_embeddings, kmeans_nodes)[0]
        if self.training:
            bipartite_edge_index, _ = dropout_edge(bipartite_edge_index, p=self.dropout)
        return bipartite_edge_index, kmean_offset_edges
        
    def _forward_per_subgraph(self, 
            x: Tensor,
            edge_index: Tensor,
            input_nodes: Tensor,
            input_edges: Tensor,
            input_edge_attr: Tensor,
            batch: Optional[Tensor] = None
        ) -> Tensor:
        """
        Forward pass for a single subgraph with proper batching.
        
        Args:
            x: Subgraph node features
            edge_index: Subgraph edge connectivity
            input_nodes: Global input node features
            input_edges: Global input edge connectivity
            input_edge_attr: Global input edge attributes
            batch: Batch assignment for the subgraph nodes
        """
        # Combine input nodes with subgraph nodes
        total_nodes = torch.cat([input_nodes, x], dim=0)
        
        # Adjust edge indices: input edges stay as-is, subgraph edges are offset
        offset_subgraph_edges = torch.add(edge_index, input_nodes.shape[0])
        combined_edge_index = torch.cat([input_edges, offset_subgraph_edges], dim=1)

        synthetic_edge_attr = torch.zeros(edge_index.shape[1], input_edge_attr.shape[1], dtype=input_edge_attr.dtype, device=input_edge_attr.device)
        combined_edge_attr = torch.cat([input_edge_attr, synthetic_edge_attr], dim=0)
        
        # Create batch tensor for the combined graph
        if batch is not None:
            num_input_nodes = input_nodes.shape[0]
            num_subgraph_nodes = x.shape[0]
            
            # Input nodes belong to all batches - we assign them to batch 0 for simplicity
            # but they will be shared across all subgraphs in the actual processing
            input_batch = torch.zeros(num_input_nodes, dtype=batch.dtype, device=batch.device)
            
            # Subgraph nodes keep their original batch assignment
            combined_batch = torch.cat([input_batch, batch], dim=0)
        else:
            combined_batch = None

        total_nodes = self.encode_conv(total_nodes, combined_edge_index, edge_attr=combined_edge_attr)

        # Encode through VGAE
        total_nodes = self.vgae(total_nodes, combined_edge_index, batch=combined_batch)[0]
    
        subgraph_nodes = total_nodes[input_nodes.shape[0]:]
        
        subgraph_nodes = self.linear_post_attention(subgraph_nodes)
        subgraph_nodes = self.norm(subgraph_nodes)
        
        if self.training and hasattr(self, 'dropout_layer'):
            subgraph_nodes = self.dropout_layer(subgraph_nodes)
        return subgraph_nodes
    
    def _forward_per_graph(self, 
            kmeans_carry: KMeansCarry,
            inputs: Data,
            batch: Optional[Tensor] = None
        ) -> Tensor:
        """
        Forward pass for graphs with iterative processing over masks.
        
        Args:
            kmeans_carry: KMeansCarry object containing clustering state
            inputs: Input graph data containing global context
            batch: Batch assignment tensor (handled by PyTorch Geometric)
            
        Returns:
            Updated node features after processing subgraphs
        """
        k = kmeans_carry.k
        if k == 0:
            return torch.zeros_like(kmeans_carry.x)
        
        # Collect features from all masks
        inputs_feats = []
        all_subgraph_features = []
        
        for mask_idx in range(k):
            # Get masked features and edges for this specific mask
            masked_x = kmeans_carry.masked_x(mask_idx)
            masked_edge_index = kmeans_carry.masked_edge_index(mask_idx)
            
            # Process through subgraph
            inputs_feats.append((masked_x, masked_edge_index, batch))

        all_subgraph_features = [self._forward_per_subgraph(x=masked_x, edge_index=masked_edges, input_nodes=inputs.x, input_edges=inputs.edge_index, input_edge_attr=inputs.edge_attr, batch=batch) for masked_x, masked_edges, batch in inputs_feats]
        
        # Stack and pool features across all subgraphs
        if all_subgraph_features:
            stacked_features = torch.stack(all_subgraph_features, dim=-1)  # [k, num_nodes, features]
            pooled_features = torch.mean(stacked_features, dim=-1)  # Average across k dimension
        else:
            pooled_features = torch.zeros_like(kmeans_carry.x)
        
        # Update kmeans_carry in place
        kmeans_carry.x = pooled_features
        
        return pooled_features
    

    def empty_carry(self, num_nodes: int, node_dim: int, device: torch.device = DEVICE, batch: Optional[Tensor] = None) -> KMeansCarry:
        """
        Create an empty KMeansCarry for initialization.
        
        Args:
            num_nodes: Number of nodes in the graph (total for all graphs if batched)
            node_dim: Dimension of node features
            device: Device to create tensors on
            batch: Batch assignment tensor if batched
            
        Returns:
            Empty KMeansCarry object
        """
        if device is None:
            device = DEVICE
            
        # Create empty tensors
        x = torch.zeros(num_nodes, node_dim, dtype=torch.float32, device=device)
        mask = torch.ones(num_nodes, self.k, dtype=torch.float32, device=device)
        none_selected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        
        return KMeansCarry(x=x, mask=mask, none_selected=none_selected, edge_index=edge_index, batch=batch)
    
    def reset_carry(self, reset_flag: torch.Tensor, carry: KMeansCarry) -> KMeansCarry:
        """
        Reset KMeansCarry based on reset flags.
        
        Args:
            reset_flag: Boolean tensor indicating which graphs to reset [batch_size]
            carry: Current KMeansCarry object
            
        Returns:
            Reset KMeansCarry object
        """
        if carry.batch is None:
            # Single graph case
            if reset_flag.item():
                # Reset to initial state
                return self.empty_carry(carry.x.shape[0], carry.x.shape[1], carry.x.device, carry.batch)
            else:
                return carry
        else:
            # Batched case
            new_x = carry.x.clone()
            new_mask = carry.mask.clone()
            new_none_selected = carry.none_selected.clone()
            
            # For each graph that needs reset, reinitialize its nodes and mask
            unique_batches = torch.unique(carry.batch)
            for i, batch_idx in enumerate(unique_batches):
                if i < len(reset_flag) and reset_flag[i]:
                    # Reset this graph
                    batch_mask = (carry.batch == batch_idx)
                    num_nodes_in_graph = batch_mask.sum()
                    
                    # Reset features
                    new_x[batch_mask] = torch.zeros_like(new_x[batch_mask])
                    new_mask[batch_mask] = torch.ones_like(new_mask[batch_mask]) / self.k
                    new_none_selected[batch_mask] = False
            
            return KMeansCarry(
                x=new_x,
                edge_index=carry.edge_index,
                batch=carry.batch,
                mask=new_mask,
                none_selected=new_none_selected
            )

    def forward(self, 
            kmeans_carry: KMeansCarry,
            inputs: Data,
            batch: Optional[Tensor] = None
        ):
        """
        Forward pass following the HierarchicalReasoningModel pattern.
        
        Args:
            kmeans_carry: KMeansCarry object containing current state
            inputs: Input graph data
            
        Returns:
            Tuple of (new_carry, output, policy_output)
        """
        # Forward iterations without gradients
        current_carry = kmeans_carry
        
        with torch.no_grad():
            for _H_step in range(self.K_cycles):
                for _L_step in range(self.L_cycles):
                    if not ((_H_step == self.K_cycles - 1) and (_L_step == self.L_cycles - 1)):
                        # Update the node features based on current clustering
                        new_features = self._forward_per_graph(current_carry, inputs, batch)
                        current_carry = KMeansCarry(
                            x=new_features,
                            edge_index=current_carry.edge_index,
                            mask=current_carry.mask,
                            none_selected=current_carry.none_selected
                        )

                if not (_H_step == self.K_cycles - 1):
                    current_carry = self.kmeans_module(current_carry, batch)
        # Assertions for debugging
        assert not current_carry.x.requires_grad, "x should not require grad (IFT)"
        assert not current_carry.mask.requires_grad, "mask should not require grad (IFT)"

        # 1-step with gradients
        new_features = self._forward_per_graph(current_carry, inputs, batch)
        current_carry = KMeansCarry(
            x=new_features,
            edge_index=current_carry.edge_index,
            mask=current_carry.mask,
            none_selected=current_carry.none_selected
        )
        current_carry = self.kmeans_module(current_carry, batch)

        # Create new carry for output
        new_carry = KMeansCarry(
            x=current_carry.x.detach(),
            edge_index=current_carry.edge_index.detach(),
            batch=current_carry.batch,
            mask=current_carry.mask.detach(),
            none_selected=current_carry.none_selected.detach()
        )
        
        # Generate outputs
        output = self.output_head(current_carry, batch)
        q_logits = self.policy_module(current_carry, batch).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

@dataclass
class KMeansHRMInitialCarry:
    inner_carry: KMeansCarry

    steps: Tensor
    halted: Tensor

    current_data: Data

class KMeansHRMOutput(TypedDict):
    y_pred: Tensor
    q_policy: Tuple[Tensor, Tensor]
    target_q_policy: Optional[Tuple[Tensor, Tensor]]


class KMeansHRMConfig(TypedDict):
    inner_module: KMeansHRMInnerModuleConfig
    explore_steps_prob: float
    halt_max_steps: int


class KMeansHRMModule(nn.Module):

    def __init__(self, config: KMeansHRMConfig, training: bool = True):
        super(KMeansHRMModule, self).__init__()
        self.config = config
        self.inner_module = KMeansHRMInnerModule(config['inner_module'], training=training)
        self.explore_steps_prob = config['explore_steps_prob']
        self.halt_max_steps = config['halt_max_steps']

        self.training = training

    def initial_carry(self, batch: Batch) -> KMeansHRMInitialCarry:
        num_nodes = batch.x.shape[0]  # Total nodes across all graphs
        node_dim = batch.x.shape[1]   # Feature dimension
        device = batch.x.device
        
        # Number of graphs in the batch
        if hasattr(batch, 'num_graphs'):
            num_graphs = batch.num_graphs
        else:
            num_graphs = len(torch.unique(batch.batch))

        initial_inner_carry = self.inner_module.empty_carry(
            num_nodes=num_nodes, 
            node_dim=node_dim, 
            device=device, 
            batch=batch.batch
        )

        # Initialize with actual data
        initial_inner_carry.x = batch.x.clone()
        initial_inner_carry.edge_index = batch.edge_index.clone()
        
        return KMeansHRMInitialCarry(
            inner_carry=initial_inner_carry,
            steps=torch.zeros(num_graphs, dtype=torch.int32, device=device),
            halted=torch.zeros(num_graphs, dtype=torch.bool, device=device),
            current_data=batch
        )


    def forward(self, 
            carry: KMeansHRMInitialCarry,
            data: Batch,
        ):
        """
        Forward pass with optional VGAE support.
        
        Returns:
            Tuple of (updated_carry, output)
        """
        new_inner_carry = self.inner_module.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        
        # For PyTorch Geometric batches, we don't need complex filtering
        # as halted/reset is handled at the graph level in reset_carry
        new_current_data = data

        new_inner_carry, y_pred, (q_halt, q_continue) = self.inner_module(new_inner_carry, new_current_data, new_current_data.batch)

        output = KMeansHRMOutput(
            y_pred=y_pred,
            q_policy=(q_halt, q_continue)
        )

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.halt_max_steps

            halted = is_last_step

            if self.training and self.halt_max_steps > 0:
                halted = halted | (q_halt > q_continue)

                min_halt_steps = (torch.rand_like(q_halt) < self.explore_steps_prob) * torch.randint_like(new_steps, low=2, high=self.halt_max_steps)
                halted = halted & (new_steps >= min_halt_steps)

                # Generate target policy for training
                target_q_logits = self.inner_module.policy_module(new_inner_carry)
                output['target_q_policy'] = (target_q_logits[..., 0], target_q_logits[..., 1])
        
        return KMeansHRMInitialCarry(inner_carry=new_inner_carry, steps=new_steps, halted=halted, current_data=new_current_data), output
    
    def reset_parameters(self):
        """Reset parameters for inner module"""
        self.inner_module.reset_parameters()
