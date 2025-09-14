from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, List, Union
import math

from torch.nn import functional as F
import torch

try:
    from torch_geometric.loader import DataLoader
except Exception:  # pragma: no cover
    DataLoader = Any  # type: ignore

import torch

PROPERTY_BOUNDS: Dict[str, Tuple[float, float]] = {}


def train_until_total_iters(
    *,
    total_iters: int,
    start_epoch: int,
    train_state: Any,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Any,
    train_batch_fn: Callable[[int, Any, Any, Any], Dict[str, float]],
    validate_fn: Optional[Callable[[int, Any, DataLoader], Dict[str, float]]] = None,
    save_checkpoint_fn: Optional[Callable[[Dict[str, Any], int, bool], None]] = None,
    checkpoint_every_n: int = 100,
    print_every_n: int = 50,
) -> Tuple[Dict[str, Any], float]:
    """
    Run training steps until reaching total_iters. Aggregates metrics per epoch and optionally validates.

    Args:
      total_iters: Absolute number of optimizer steps to run (global across epochs).
      start_epoch: Starting epoch index (1-based) for logging.
      train_state: Object with attributes: model, optimizer, step, total_steps, carry.
      train_loader: Training DataLoader yielding PyG Batch objects.
      val_loader: Optional validation DataLoader.
      config: Training config (any mapping-like object).
      train_batch_fn: Function performing one training step. Signature: (epoch, train_state, batch, config) -> metrics dict.
      validate_fn: Optional validation function. Signature: (epoch, model, loader) -> metrics dict.
      save_checkpoint_fn: Optional callback to persist checkpoints. Called as (state_dict_like, step, is_best).
      checkpoint_every_n: Save a checkpoint every N steps.
      print_every_n: Print running status every N steps.

    Returns:
      (history, best_val_loss)
    """
    model = train_state.model

    history: Dict[str, Any] = {"train": [], "val": []}
    best_val_loss = float("inf")

    epoch = max(1, int(start_epoch))
    # Continue until global steps reach target
    while train_state.step < total_iters:
        running_loss = 0.0
        mae_accum: Dict[str, float] = {}
        count_samples = 0

        # Train for one epoch (or partial if we reach total_iters)
        for batch_idx, batch_data in enumerate(train_loader, start=1):
            if train_state.step >= total_iters:
                break

            metrics = train_batch_fn(epoch, train_state, batch_data, config)

            batch_size_effective = getattr(batch_data, "num_graphs", None)
            if batch_size_effective is None:
                # Fallback: try len of list conversion (PyG Batch supports to_data_list)
                try:
                    batch_size_effective = len(batch_data.to_data_list())  # type: ignore[attr-defined]
                except Exception:
                    batch_size_effective = 1

            running_loss += float(metrics.get("loss", 0.0)) * batch_size_effective
            # Lazily initialize MAE keys to accumulate properly
            for k, v in list(metrics.items()):
                if k.startswith("mae_") and not math.isnan(float(v)):
                    mae_accum[k] = mae_accum.get(k, 0.0) + float(v) * batch_size_effective
            count_samples += batch_size_effective

            if print_every_n and (train_state.step % print_every_n == 0):
                try:
                    lr = max(g.get("lr", None) for g in train_state.optimizer.param_groups)  # type: ignore[attr-defined]
                except Exception:
                    lr = None
                print(
                    f"step {train_state.step:>6}/{total_iters} | epoch {epoch:>3} | batch {batch_idx:>4} | "
                    f"loss={metrics.get('loss', float('nan')):.4f} | lr={lr if lr is not None else 'n/a'}"
                )

            # Periodic checkpoint by global step
            if save_checkpoint_fn is not None and checkpoint_every_n > 0 and train_state.step % checkpoint_every_n == 0:
                to_save = {
                    "epoch": epoch,
                    "global_step": train_state.step,
                    "model_state": model.state_dict(),
                    "optimizer_state": train_state.optimizer.state_dict(),
                    "model_class": model.__class__.__name__,
                    "model_module": model.__class__.__module__,
                    "train_state": {
                        "step": int(train_state.step),
                        "total_steps": int(train_state.total_steps),
                        "carry": train_state.carry,
                    },
                }
                save_checkpoint_fn(to_save, step=train_state.step, is_best=False)

        # End epoch aggregation
        if count_samples > 0:
            train_stats = {
                "epoch": epoch,
                "loss": running_loss / max(1, count_samples),
                **{k: (v / max(1, count_samples)) for k, v in mae_accum.items()},
            }
        else:
            train_stats = {"epoch": epoch, "loss": float("nan")}

        val_stats: Dict[str, Any] = {"epoch": epoch}
        if validate_fn is not None and val_loader is not None:
            val_out = validate_fn(epoch, model, val_loader)
            val_stats.update(val_out)
            is_best = float(val_out.get("loss", float("inf"))) < best_val_loss
            if is_best:
                best_val_loss = float(val_out["loss"])  # type: ignore[index]

            if save_checkpoint_fn is not None:
                to_save = {
                    "epoch": epoch,
                    "global_step": train_state.step,
                    "model_state": model.state_dict(),
                    "optimizer_state": train_state.optimizer.state_dict(),
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                    "model_class": model.__class__.__name__,
                    "model_module": model.__class__.__module__,
                    "train_state": {
                        "step": int(train_state.step),
                        "total_steps": int(train_state.total_steps),
                        "carry": train_state.carry,
                    },
                }
                save_checkpoint_fn(to_save, step=epoch, is_best=is_best)

        history["train"].append(train_stats)
        history["val"].append(val_stats)

        print(
            f"[total_iters] Epoch {epoch:03d} done | train_loss={train_stats.get('loss', float('nan')):.4f} | "
            f"val_loss={val_stats.get('loss', float('nan')):.4f} | step={train_state.step}/{total_iters}"
        )

        epoch += 1

    return history, best_val_loss


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str | torch.device] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Load a checkpoint and restore the model (and optionally optimizer).

    Returns (raw_state_dict, saved_train_state_dict_or_none)
    """
    if map_location is None:
        map_location = "cpu"
    state = torch.load(checkpoint_path, map_location=map_location)

    model_state = state.get("model_state")
    if model_state is None:
        raise ValueError("Checkpoint missing 'model_state'")
    model.load_state_dict(model_state, strict=False)

    if optimizer is not None and "optimizer_state" in state:
        try:
            optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[index]
        except Exception:
            # Optimizer states can be shape-dependent; ignore if incompatible
            pass

    return state, state.get("train_state")  # type: ignore[return-value]


def init_property_bounds(properties: List[str], extended_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    global PROPERTY_BOUNDS
    for prop in properties:
        values = extended_df[prop].dropna()
        if len(values) > 0:
            min_val = float(values.min())
            max_val = float(values.max())
            PROPERTY_BOUNDS[prop] = (min_val, max_val)
        else:
            # Fallback to a default range if no values are present
            PROPERTY_BOUNDS[prop] = (0.0, 1.0)

def range_violation_loss(properties: List[str], pred: torch.Tensor, prop_idx: int) -> torch.Tensor:
    global PROPERTY_BOUNDS
    bounds_tensor = torch.tensor([PROPERTY_BOUNDS[p] for p in properties], dtype=torch.float32)  # [5,2]

    lo, hi = bounds_tensor[prop_idx, 0], bounds_tensor[prop_idx, 1]
    below = F.relu(lo - pred)
    above = F.relu(pred - hi)
    return torch.pow(below + above, 2)

def smiles_weight(monomer_index: int, aux_info: Union[List[np.ndarray], np.ndarray], weight_base: float = 0.33) -> torch.Tensor:

    if isinstance(aux_info, list):
        monomer_count = torch.tensor([aux_info[i][monomer_index] for i in range(len(aux_info))], dtype=torch.float32)
    else:
        monomer_count = torch.tensor(aux_info[monomer_index], dtype=torch.float32)

    return torch.pow(weight_base, monomer_count-1)

def composite_loss(monomer_index: int, properties: List[str], preds: torch.Tensor, targets: torch.Tensor, related_info: Union[List[np.ndarray], np.ndarray]) -> torch.Tensor:
    # preds: [B, 5], targets: [B, 5] with NaNs if missing
    loss_items = []
    for j in range(len(properties)):
        t = targets[:, j]
        p = preds[:, j]
        mask_present = torch.isfinite(t)

        if mask_present.any():
            mse = F.mse_loss(p[mask_present], t[mask_present], weight=smiles_weight(monomer_index, related_info)[mask_present])
            loss_items.append(mse)
        # Range-violation for all (including available): no loss if within bounds
        rv = range_violation_loss(properties, p, j).mean()
        loss_items.append(rv)
    return torch.stack(loss_items).mean()


@torch.no_grad()
def compute_mae_in_bounds(monomer_index: int, properties: List[str], preds: torch.Tensor, targets: torch.Tensor, related_info: np.ndarray) -> Dict[str, float]:
    out = {}
    for j, name in enumerate(properties):

        t = targets[:, j]
        p = preds[:, j]
        mask_present = torch.isfinite(t)

        if mask_present.any():
            out[f"mae_{name}"] = (torch.mul(p[mask_present] - t[mask_present], smiles_weight(monomer_index, related_info)[mask_present])).abs().mean().item()
        else:
            out[f"mae_{name}"] = float("nan")
    return out

print("Loss and metrics initialized.")