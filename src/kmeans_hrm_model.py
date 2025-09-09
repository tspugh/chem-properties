from dataclasses import dataclass
from typing import List, Union, Tuple, Optional, Dict, TypedDict, Literal, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    MessagePassing, 
    GATConv, 
    GCNConv,
    VGAE,
    InnerProductDecoder,
    LayerNorm,
    global_add_pool,
    global_max_pool,
    GlobalAttention,
    global_mean_pool,
    ChebConv,
)
from torch_geometric.nn.pool import radius
from torch_geometric.utils import (
    add_self_loops, 
    subgraph,
    bipartite_subgraph,
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

# HEAVILY REFERENCES https://github.com/sapientinc/HRM/blob/05dd4ef795a98c20110e380a330d0b3ec159a46b/models/hrm/hrm_act_v1.py#L222

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class KMeansCarry:
    nodes: Tensor
    mask: Tensor
    none_selected: Tensor
    edge_index: Tensor
    
    @property
    def expanded_weights(self) -> Tensor:
        return self.nodes.unsqueeze(-1)

    def as_list(self) -> List['KMeansCarry']:
        # Use torch.unbind for parallel processing instead of list comprehension
        mask_list = torch.unbind(self.mask, dim=1)
        return [KMeansCarry(nodes=self.nodes, mask=mask, none_selected=self.none_selected, edge_index=self.edge_index) for mask in mask_list]
    
    def get_subgraphs(self, include_none_selected: bool = False) -> Tuple[Tensor, Tensor]:
        if include_none_selected:
            edge_indices, _ = torch.vmap(bipartite_subgraph, dim=0)(torch.cat([self.mask, self.none_selected], dim=1), self.edge_index.unsqueeze(0))
        else:
            edge_indices, _ = torch.vmap(bipartite_subgraph, dim=0)(self.mask, self.edge_index.unsqueeze(0))
        return (self.nodes.unsqueeze(0), edge_indices)

# Takes in a mask and feature nodes, outputs a new mask
class KMeansHeadConfig(TypedDict):
    node_count: int
    node_dim: int
    max_nodes: int

    num_layers: int
    dropout: float

    # In: mask, features; Out: weighted_features
    weighting_module: nn.Module
    # In: mask, weighted_features; Out: center_mask
    center_module: nn.Module
    # In: center_mask, weighted_features; Out: mask
    mask_module: nn.Module

    act: str
    norm: str
    norm_kwargs: Dict[str, Any]
    act_kwargs: Dict[str, Any]


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
        self.norm = normalization_resolver(config["norm"], **config["norm_kwargs"])


    def forward(self, k_data: KMeansCarry, batch_size: Optional[int] = None):
        """
        Forward pass for KMeansHead.
        
        Args:
            k_data: KMeansCarry object containing nodes, mask, none_selected, and edge_index
            batch_size: Optional batch size. If provided and positive, assumes batching along dimension 0
            
        Returns:
            KMeansCarry object with updated mask
        """
        # Handle batching: if batch_size is provided and positive, process each batch separately
        if batch_size is not None and batch_size > 0:
            # Reshape for batch processing
            nodes = k_data.nodes.view(batch_size, -1, k_data.nodes.shape[-1])
            mask = k_data.mask.view(batch_size, -1, k_data.mask.shape[-1])
            none_selected = k_data.none_selected.view(batch_size, -1)
            
            # Use vmap for parallel processing across batches
            def process_batch(batch_nodes, batch_mask, batch_none_selected):
                batch_k_data = KMeansCarry(
                    nodes=batch_nodes,
                    mask=batch_mask,
                    none_selected=batch_none_selected,
                    edge_index=k_data.edge_index
                )
                return self._forward_single(batch_k_data)
            
            # Apply vmap for parallel processing
            batch_outputs = torch.vmap(process_batch)(nodes, mask, none_selected)
            
            # Extract results from batched outputs
            output_nodes = torch.stack([out.nodes for out in batch_outputs], dim=0)
            output_mask = torch.stack([out.mask for out in batch_outputs], dim=0)
            output_none_selected = torch.stack([out.none_selected for out in batch_outputs], dim=0)
            
            return KMeansCarry(
                nodes=output_nodes,
                mask=output_mask,
                none_selected=output_none_selected,
                edge_index=k_data.edge_index
            )
        else:
            return self._forward_single(k_data)
    
    def _forward_single(self, k_data: KMeansCarry):
        """
        Forward pass for a single sample (no batching).
        
        Args:
            k_data: KMeansCarry object containing nodes, mask, none_selected, and edge_index
            
        Returns:
            KMeansCarry object with updated mask
        """
        masked_features = k_data.nodes * k_data.mask
        
        weighted_features = self.weighting_module(masked_features, k_data.edge_index)
        center_mask = self.center_module(weighted_features, k_data.edge_index)
        output_mask = self.mask_module(center_mask, weighted_features, k_data.edge_index)
        output_mask = self.norm(output_mask)

        if self.act is not None:
            output_mask = self.act(output_mask)
        if self.training and hasattr(self, 'dropout') and self.dropout is not None:
            output_mask = self.dropout(output_mask)

        return KMeansCarry(nodes=k_data.nodes, mask=output_mask, none_selected=k_data.none_selected, edge_index=k_data.edge_index)
    
    def reset_parameters(self):
        """Reset parameters for all modules in KMeansHead"""
        if hasattr(self.weighting_module, 'reset_parameters'):
            self.weighting_module.reset_parameters()
        if hasattr(self.center_module, 'reset_parameters'):
            self.center_module.reset_parameters()
        if hasattr(self.mask_module, 'reset_parameters'):
            self.mask_module.reset_parameters()
        if hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()


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
        self.dropouts = nn.ModuleList()
        
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
            self.dropouts.append(nn.Dropout(config["dropout"]))
    
    def forward(self, features: Tensor, edge_index: Tensor, batch_size: Optional[int] = None) -> Tensor:
        """
        Forward pass with mask-based filtering through multiple spectral layers.
        
        Args:
            features: Node features tensor
            edge_index: Edge connectivity [2, num_edges]
            batch_size: Optional batch size. If provided and positive, assumes batching along dimension 0
            
        Returns:
            Weighted node features after processing through all layers
        """
        # Handle batching: if batch_size is provided and positive, process each batch separately
        if batch_size is not None and batch_size > 0:
            # Reshape for batch processing
            features_batched = features.view(batch_size, -1, features.shape[-1])
            
            # Use vmap for parallel processing across batches
            def process_batch(batch_features):
                return self._forward_single(batch_features, edge_index)
            
            # Apply vmap for parallel processing
            return torch.vmap(process_batch)(features_batched)
        else:
            return self._forward_single(features, edge_index)
    
    def _forward_single(self, features: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass for a single sample (no batching) through multiple spectral layers.
        
        Args:
            features: Node features tensor
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Weighted node features after processing through all layers
        """
        x = features
        
        # Process through each layer
        for i in range(self.num_layers):
            # Apply Chebyshev spectral convolution
            x = self.cheb_convs[i](
                x=features,
                edge_index=edge_index,
                edge_weight=None,
                batch=None,
                lambda_max=None
            )
            
            if self.norms[i] is not None:
                x = self.norms[i](x)
            if self.training and self.dropouts[i] is not None:
                x = self.dropouts[i](x)
        
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
    
    def forward(self, weighted_features: Tensor) -> Tensor:
        """
        Calculate weighted mean and return mask for closest node.
        
        Args:
            mask: Binary mask tensor (1 for nodes to include, 0 for nodes to exclude)
            weighted_features: PyTorch Geometric Batch object containing weighted node features
            
        Returns:
            New mask tensor with closest node to mean set to 1, rest to 0
        """
        # Calculate weighted mean (sum of masked features / sum of mask)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        mask_sum = torch.sum(weighted_features) + epsilon
        center = torch.sum(weighted_features, dim=0) / mask_sum
        
        # Calculate distances from all nodes to the center
        distances = self.distance_module(weighted_features, center)
        
        # Find the node with minimum distance
        closest_node_idx = torch.argmin(distances)
        
        # Use torch.zeros, to ensure the tensor is on the same device and with the same type as mask
        center_mask = torch.zeros(weighted_features.size(), dtype=torch.bool, device=weighted_features.device)
        center_mask[closest_node_idx] = 1
        
        return center_mask
    


class RadiusMaskConfig(TypedDict):
    k: int
    radius: float
    weighting_module: nn.Module


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
                k: Maximum number of nodes to include in the mask
                radius: Radius within which to search for nodes
                weighting_module: Module to weight the features
        """
        super(RadiusAttentionWeights, self).__init__()
        self.k = config["k"]
        self.radius = config["radius"]

        self.weighting_module = config["weighting_module"]
    
    def forward(self, center_mask: Tensor, features: Tensor, edge_index: Tensor, batch_size: Optional[int] = None) -> Tensor:
        """
        Find nodes within radius from center nodes and return a new mask.
        
        Args:
            center_mask: Binary mask tensor indicating center nodes (1 for center, 0 otherwise)
            features: Node features tensor
            edge_index: Edge connectivity tensor
            batch_size: Optional batch size. If provided and positive, assumes batching along dimension 0
            
        Returns:
            New mask tensor with nodes within radius set to 1, rest to 0
        """
        # Handle batching: if batch_size is provided and positive, process each batch separately
        if batch_size is not None and batch_size > 0:
            # Reshape for batch processing
            center_mask_batched = center_mask.view(batch_size, -1)
            features_batched = features.view(batch_size, -1, features.shape[-1])
            
            # Use vmap for parallel processing across batches
            def process_batch(batch_center_mask, batch_features):
                return self._forward_single(batch_center_mask, batch_features, edge_index)
            
            # Apply vmap for parallel processing
            return torch.vmap(process_batch)(center_mask_batched, features_batched)
        else:
            return self._forward_single(center_mask, features, edge_index)
    
    def _forward_single(self, center_mask: Tensor, features: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass for a single sample (no batching).
        
        Args:
            center_mask: Binary mask tensor indicating center nodes (1 for center, 0 otherwise)
            features: Node features tensor
            edge_index: Edge connectivity tensor
            
        Returns:
            New mask tensor with nodes within radius set to 1, rest to 0
        """
        center_features = features[center_mask]
        
        # Use radius function to find neighbors
        # x: all node features, y: center node features
        assign_index = radius(features, center_features, self.radius, max_num_neighbors=self.k)
        
        # Create new mask
        new_mask = torch.zeros(center_mask.size(), dtype=center_mask.dtype, device=center_mask.device)
        
        neighbor_indices = assign_index[0]
        
        new_mask = self.weighting_module(features[neighbor_indices])
        
        return new_mask
    
    def reset_parameters(self):
        """Reset parameters for weighting module"""
        if hasattr(self.weighting_module, 'reset_parameters'):
            self.weighting_module.reset_parameters()

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

    def forward(self, kmeans_carry: KMeansCarry, batch_size: Optional[int] = None) -> KMeansCarry:
        """
        Args:
            kmeans_carry: KMeansCarry object containing nodes, mask, none_selected, and edge_index
            batch_size: Optional batch size. If provided and positive, assumes batching along dimension 0
        Returns:
            KMeansCarry object with updated mask
        """
        # Handle batching: if batch_size is provided and positive, process each batch separately
        if batch_size is not None and batch_size > 0:
            # Reshape for batch processing
            nodes = kmeans_carry.nodes.view(batch_size, -1, kmeans_carry.nodes.shape[-1])
            mask = kmeans_carry.mask.view(batch_size, -1, kmeans_carry.mask.shape[-1])
            none_selected = kmeans_carry.none_selected.view(batch_size, -1)
            
            # Use vmap for parallel processing across batches
            def process_batch(batch_nodes, batch_mask, batch_none_selected):
                batch_kmeans_carry = KMeansCarry(
                    nodes=batch_nodes,
                    mask=batch_mask,
                    none_selected=batch_none_selected,
                    edge_index=kmeans_carry.edge_index
                )
                return self._forward_single(batch_kmeans_carry)
            
            # Apply vmap for parallel processing
            batch_outputs = torch.vmap(process_batch)(nodes, mask, none_selected)
            
            # Extract results from batched outputs
            output_nodes = torch.stack([out.nodes for out in batch_outputs], dim=0)
            output_mask = torch.stack([out.mask for out in batch_outputs], dim=0)
            output_none_selected = torch.stack([out.none_selected for out in batch_outputs], dim=0)
            
            return KMeansCarry(
                nodes=output_nodes,
                mask=output_mask,
                none_selected=output_none_selected,
                edge_index=kmeans_carry.edge_index
            )
        else:
            return self._forward_single(kmeans_carry)
    
    def _forward_single(self, kmeans_carry: KMeansCarry) -> KMeansCarry:
        """
        Forward pass for a single sample (no batching).
        
        Args:
            kmeans_carry: KMeansCarry object containing nodes, mask, none_selected, and edge_index
        Returns:
            KMeansCarry object with updated mask
        """
        num_nodes = kmeans_carry.nodes.shape[1]
        
        masks_i = [head(k_carry) for head, k_carry in zip(self.heads, kmeans_carry.as_list())]
        mask_tensor = torch.stack(masks_i, dim=1)  # [num_nodes, k]

        # Only consider weights strictly greater than threshold
        above_thresh = torch.greater(mask_tensor, self.thresh)
        masked_weights = kmeans_carry.nodes.unsqueeze(-1).masked_fill(~above_thresh, float("-inf"))

        top_vals, top_idx = torch.topk(masked_weights, k=self.max_overlap, dim=1)
        valid = torch.isfinite(top_vals)

        row_idx = torch.arange(num_nodes, device=kmeans_carry.nodes.device).unsqueeze(1).expand_as(top_idx)
        row_idx_valid = row_idx[valid]
        col_idx_valid = top_idx[valid]
        mask_tensor[row_idx_valid, col_idx_valid] = 1

        none_selected = torch.zeros(num_nodes, device=kmeans_carry.nodes.device, dtype=mask_tensor.dtype)
        # Optional extra cluster for nodes not assigned to any mask
        if self.excluded_is_cluster:
            none_selected = mask_tensor.sum(dim=1) == 0

        return KMeansCarry(nodes=kmeans_carry.nodes, mask=mask_tensor, none_selected=none_selected, edge_index=kmeans_carry.edge_index)





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
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, x: Tensor, edge_index: Tensor, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass returning mu and logstd
        
        Args:
            x: Node features tensor
            edge_index: Edge connectivity tensor
            batch_size: Optional batch size. If provided and positive, assumes batching along dimension 0
        """
        # Handle batching: if batch_size is provided and positive, process each batch separately
        if batch_size is not None and batch_size > 0:
            # Reshape for batch processing
            x_batched = x.view(batch_size, -1, x.shape[-1])
            
            # Use vmap for parallel processing across batches
            def process_batch(batch_x):
                return self._forward_single(batch_x, edge_index)
            
            # Apply vmap for parallel processing
            batch_results = torch.vmap(process_batch)(x_batched)
            
            # Extract mu and logstd from results
            mu = torch.stack([result[0] for result in batch_results], dim=0)
            logstd = torch.stack([result[1] for result in batch_results], dim=0)
            
            return mu, logstd
        else:
            return self._forward_single(x, edge_index)
    
    def _forward_single(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass for a single sample (no batching) returning mu and logstd"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer for mean (mu)
        mu = self.convs[-1](x, edge_index)
        
        # For logstd, we use the same architecture but with different weights
        # In practice, you might want to share some layers between mu and logstd
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
        
        # Linear layers
        self.linear1 = nn.Linear(self.node_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Normalization and activation
        self.norm = normalization_resolver(config["norm"], **config["norm_kwargs"])
        self.act = activation_resolver(config["act"], **config["act_kwargs"])
    
    def forward(self, kmeans_carry: KMeansCarry, batch_size: Optional[int] = None) -> Tensor:
        """
        Forward pass for output head.
        
        Args:
            kmeans_carry: KMeansCarry object containing nodes and edge_index
            
        Returns:
            Output tensor of shape [output_dim]
        """
        nodes = kmeans_carry.nodes
        edge_index = kmeans_carry.edge_index
        
        # Apply graph pooling
        if self.pooling_type == "attention":
            pooled = self.pooling(nodes, batch=None)
        else:
            pooled = self.pooling(nodes, batch=None)
        
        # Apply linear layers with normalization and activation
        x = self.linear1(pooled)
        if self.norm is not None:
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


    vgae_encoder_type: str = "gcn"  # "gcn", "gat", "custom"
    vgae_latent_dim: int = 64
    vgae_encoder_layers: int = 2
    vgae_encoder_dropout: float = 0.1
    vgae_decoder_type: Optional[str] = None  # None for InnerProductDecoder, "custom" for custom decoder
    vgae_kl_weight: float = 1.0  # Weight for KL divergence loss

    



class KMeansHRMInnerModule(nn.Module):

    def __init__(self, config: KMeansHRMInnerModuleConfig, training: bool = True):
        super(KMeansHRMInnerModule, self).__init__()
        self.config = config
        self.add_self_loops = self.config['add_self_loops']
        self.dropout = self.config['dropout']
        self.hidden_dim = self.config['hidden_dim']
        self.node_dim = self.config['node_dim']
        self.layers = self.config['layers']

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

        # Initialize VGAE if configured
        self.vgae_encoder = VGAEEncoder(
            input_dim=self.node_dim,  # Assuming node features dimension
            hidden_dim=self.hidden_dim,
            latent_dim=self.config['vgae_latent_dim'],
            layers=self.config['vgae_encoder_layers'],
            dropout=self.config['vgae_encoder_dropout'] if training else None,
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
            
        # Standard attention layers (GAT)
        self.attention_layers = nn.ModuleList()
        for _ in range(self.layers):
            self.attention_layers.append(
                ChebConv(
                    in_channels=self.node_dim,
                    out_channels=self.node_dim//8,
                    heads=8,
                    dropout=self.dropout if training else None
                )
            )
        
        self.linear_post_attention = nn.Sequential(
            nn.Linear((self.node_dim//8)*8, self.node_dim*8),
            nn.ReLU(),
            (nn.Dropout(self.dropout) if training else None),
            nn.Linear(self.node_dim*8, self.node_dim),
            nn.LayerNorm(self.node_dim)
        )

        channels = self.k
        if self.excluded_is_cluster:
            channels += 1
        
        self.mini_attention_pool = nn.Sequential(
            nn.Linear(channels, self.hidden_dim),
            nn.ReLU(),
            (nn.Dropout(self.dropout) if training else None),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.norm = LayerNorm(self.node_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

        self.output_head = OutputHead(self.output_head_config, training=training)
        self.policy_module = OutputHead(self.policy_module_config, training=training)

    def reset_parameters(self):
        """Reset parameters for all modules in KMeansHRMInnerModule"""
        self.kmeans_module.reset_parameters()
        self.vgae_encoder.reset_parameters()
        self.vgae.reset_parameters()
        for layer in self.attention_layers:
            layer.reset_parameters()
        self.linear_post_attention.reset_parameters()
        self.norm.reset_parameters()
        self.dropout_layer.reset_parameters()
        self.mini_attention_pool.reset_parameters()
        self.output_head.reset_parameters()
        self.policy_module.reset_parameters()
        
        # Reset input embeddings if they exist
        if hasattr(self, 'input_embedding'):
            self.input_embedding.reset_parameters()
        if hasattr(self, 'puzzle_embedding'):
            self.puzzle_embedding.reset_parameters()
    


    def _mini_attention_pool(self, nodes: Tensor) -> Tensor:
        per_feature_attention = torch.vmap(self.mini_attention_pool, dim=0)(nodes)
        pool_filtered_nodes = torch.sum(torch.where(per_feature_attention > 0.5, nodes, 0), dim=0)
        pool_filtered_nodes = self.norm(pool_filtered_nodes)
        return pool_filtered_nodes
    
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
        bipartite_edge_index, _ = bipartite_subgraph(input_embeddings, kmeans_nodes)
        if self.training:
            bipartite_edge_index, _ = dropout_edge(bipartite_edge_index, p=self.dropout)
        return bipartite_edge_index, kmean_offset_edges
    
    def _forward_per_subgraph(self,
            input_nodes: Tensor,
            input_edge_index: Tensor,
            kmeans_nodes: Tensor,
            kmeans_edge_index: Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        linked_edge_index, new_kmeans_edges = self._link_subgraph_to_nodes(input_nodes, kmeans_nodes, kmeans_edge_index)
        if self.add_self_loops:
            linked_edge_index, _ = add_self_loops(linked_edge_index)
        if self.training:
            linked_edge_index, _ = dropout_edge(linked_edge_index, p=self.dropout)

        total_nodes = torch.cat([input_nodes, kmeans_nodes], dim=0)
        total_edges = torch.cat([input_edge_index, new_kmeans_edges, linked_edge_index], dim=1)

        new_nodes, new_edges = self.vgae(total_nodes, total_edges)
        new_nodes = new_nodes[input_nodes.shape[0]:]
        new_edges, _ = subgraph(torch.arange(input_nodes.shape[0], total_nodes.shape[0]), new_edges)

        # Apply attention layers
        for layer in self.attention_layers:
            new_nodes = layer(new_nodes, new_edges)
        new_nodes = self.linear_post_attention(new_nodes)

        new_nodes = self.norm(new_nodes)
        if self.dropout and self.training:
            new_nodes = self.dropout(new_nodes)
        return new_nodes, new_edges, new_kmeans_edges

    def _forward_per_graph(self, 
            kmeans_carry: KMeansCarry,
            input_nodes: Tensor,
            input_edges: Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single graph.
        """
        subgraphs, subgraphs_edge_indices = kmeans_carry.get_subgraphs(include_none_selected=self.excluded_is_cluster)

        # Use vmap for parallel processing across subgraphs
        new_node_sets = torch.vmap(self._forward_per_subgraph, dim=0)(
            subgraphs, 
            subgraphs_edge_indices, 
            input_nodes.unsqueeze(0), 
            input_edges.unsqueeze(0)
        )

        new_node_set = torch.vmap(self._mini_attention_pool, dim=1)(new_node_sets)
        return new_node_set

    def empty_carry(self, num_nodes: int, node_dim: int, device: torch.device = DEVICE, batch_size: Optional[int] = None) -> KMeansCarry:
        """
        Create an empty KMeansCarry for initialization.
        
        Args:
            batch_size: Number of batches
            num_nodes: Number of nodes in the graph
            node_dim: Dimension of node features
            device: Device to create tensors on
            
        Returns:
            Empty KMeansCarry object
        """
        if device is None:
            device = DEVICE

        if batch_size is None or batch_size <= 0:
            batch_size = 1
            
        # Create empty tensors
        nodes = torch.empty(batch_size, num_nodes, node_dim, dtype=torch.float32, device=device)
        mask = torch.zeros(batch_size, num_nodes, self.k, dtype=torch.float32, device=device)
        none_selected = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        edge_index = torch.empty(batch_size, 2, 0, dtype=torch.long, device=device)
        
        return KMeansCarry(nodes=nodes, mask=mask, none_selected=none_selected, edge_index=edge_index)
    
    def reset_carry(self, reset_flag: torch.Tensor, carry: KMeansCarry) -> KMeansCarry:
        """
        Reset KMeansCarry based on reset flags.
        
        Args:
            reset_flag: Boolean tensor indicating which samples to reset [batch_size]
            carry: Current KMeansCarry object
            
        Returns:
            Reset KMeansCarry object
        """
        # Create initial values for reset
        init_nodes = torch.zeros_like(carry.nodes)
        init_mask = torch.zeros_like(carry.mask)
        init_none_selected = torch.zeros_like(carry.none_selected)
        init_edge_index = torch.empty_like(carry.edge_index)
        
        # Apply reset flag
        reset_flag_nodes = reset_flag.view(-1, 1, 1, 1)
        reset_flag_mask = reset_flag.view(-1, 1, 1)
        reset_flag_none = reset_flag.view(-1, 1)
        reset_flag_edge_index = reset_flag.view(-1, 1, 1)
        
        new_nodes = torch.where(reset_flag_nodes, init_nodes, carry.nodes)
        new_mask = torch.where(reset_flag_mask, init_mask, carry.mask)
        new_none_selected = torch.where(reset_flag_none, init_none_selected, carry.none_selected)
        new_edge_index = torch.where(reset_flag_edge_index, init_edge_index, carry.edge_index)
        
        return KMeansCarry(
            nodes=new_nodes,
            mask=new_mask,
            none_selected=new_none_selected,
            edge_index=new_edge_index
        )

    def forward(self, 
            kmeans_carry: KMeansCarry,
            inputs: Data,
            batch_dim: Optional[int] = 0
        ):
        """
        Forward pass following the HierarchicalReasoningModel pattern.
        
        Args:
            kmeans_carry: KMeansCarry object containing current state
            inputs: Input graph data
            batch_size: Batch size
            
        Returns:
            Tuple of (output, policy_output)
        """
        # Forward iterations
        with torch.no_grad():
            for _H_step in range(self.K_cycles):
                for _L_step in range(self.L_cycles):
                    if not ((_H_step == self.K_cycles - 1) and (_L_step == self.L_cycles - 1)):
                        kmeans_carry.nodes = self._forward_per_graph(kmeans_carry, inputs, inputs.edge_index)

                if not (_H_step == self.K_cycles - 1):
                    kmeans_carry.mask = self.kmeans_module(kmeans_carry)

        assert not kmeans_carry.nodes.requires_grad and not kmeans_carry.mask.requires_grad

        # 1-step grad
        kmeans_carry.nodes = self._forward_per_graph(kmeans_carry, inputs.x, inputs.edge_index)
        kmeans_carry.mask = self.kmeans_module(kmeans_carry)

        # Create new carry
        new_carry = KMeansCarry(
            nodes=kmeans_carry.nodes.detach(),
            mask=kmeans_carry.mask.detach(),
            none_selected=kmeans_carry.none_selected.detach(),
            edge_index=kmeans_carry.edge_index.detach()
        )

        # LM Outputs
        output = self.output_head(new_carry)

        # Q head (policy module)
        q_logits = self.policy_module(new_carry).to(torch.float32)
        
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

    def initial_carry(self, batch: Data) -> KMeansCarry:
        batch_size = batch.x.shape[0]
        return KMeansHRMInitialCarry(
            inner_carry=self.inner_module.empty_carry(batch_size),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=DEVICE),
            halted=torch.zeros(batch_size, dtype=torch.int32, device=DEVICE),
            current_data=batch
        )


    def forward(self, 
            carry: KMeansHRMInitialCarry,
            data: Data,
        ) -> KMeansHRMOutput:
        """
        Forward pass with optional VGAE support.
        
        Returns:
            Tuple of (updated_nodes, updated_edges)
        """

        new_inner_carry = self.inner_module.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        
        def create_filtered_data(new_data: Data) -> Data:
            """Create new Data object with halted entries preserving old values"""
            for key, new_value in new_data.__dict__.items():
                if isinstance(new_value, Tensor) and hasattr(carry.current_data, key):
                    old_value = getattr(carry.current_data, key)
                    halt_mask = carry.halted.view((-1,))
                    
                    new_data[key] = torch.where(halt_mask, old_value, new_value)
                else:
                    new_data[key] = new_value
            
            return new_data
        
        new_current_data = create_filtered_data(data)

        new_inner_carry, y_pred, (q_halt, q_continue) = self.inner_module(new_inner_carry, new_current_data)

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

                output['target_q_policy'] = self.inner_module.policy_module(new_inner_carry.nodes, new_inner_carry.edge_index)[-1]
        
        return KMeansHRMInitialCarry(inner_carry=new_inner_carry, steps=new_steps, halted=halted, current_data=new_current_data), output
    
    def reset_parameters(self):
        """Reset parameters for inner module"""
        self.inner_module.reset_parameters()
