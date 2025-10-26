"""
Utility functions for tensor/numpy conversions and device validation.

These utilities help ensure consistent handling of PyTorch tensors and NumPy arrays
across the codebase, preventing common device mismatch and conversion errors.
"""

import numpy as np
from typing import Union, List, Optional
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def to_numpy(data: Union['torch.Tensor', np.ndarray, List]) -> np.ndarray:
    """
    Convert tensor or list to numpy array.

    Args:
        data: PyTorch tensor, numpy array, or list

    Returns:
        numpy array

    Example:
        >>> tensor = torch.tensor([1, 2, 3])
        >>> arr = to_numpy(tensor)
        >>> isinstance(arr, np.ndarray)
        True
    """
    if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    else:
        # Try to convert to numpy
        try:
            return np.array(data)
        except Exception as e:
            raise TypeError(f"Cannot convert type {type(data)} to numpy array: {e}")


def to_tensor(data: Union['torch.Tensor', np.ndarray, List],
              dtype: Optional['torch.dtype'] = None,
              device: Optional[Union[str, 'torch.device']] = None) -> 'torch.Tensor':
    """
    Convert numpy array or list to PyTorch tensor.

    Args:
        data: PyTorch tensor, numpy array, or list
        dtype: Target dtype (default: infer from data)
        device: Target device (default: CPU)

    Returns:
        PyTorch tensor

    Example:
        >>> arr = np.array([1, 2, 3])
        >>> tensor = to_tensor(arr, device='cpu')
        >>> isinstance(tensor, torch.Tensor)
        True
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")

    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        tensor = torch.tensor(data)
    else:
        try:
            tensor = torch.tensor(data)
        except Exception as e:
            raise TypeError(f"Cannot convert type {type(data)} to tensor: {e}")

    # Convert dtype if specified
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)

    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def ensure_device(tensor: 'torch.Tensor',
                  device: Union[str, 'torch.device'],
                  warn_on_move: bool = False) -> 'torch.Tensor':
    """
    Ensure tensor is on the specified device.

    Args:
        tensor: PyTorch tensor
        device: Target device
        warn_on_move: Whether to warn when moving tensor between devices

    Returns:
        Tensor on specified device

    Example:
        >>> tensor = torch.tensor([1, 2, 3])
        >>> tensor = ensure_device(tensor, 'cpu')
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    target_device = torch.device(device)

    if tensor.device != target_device:
        if warn_on_move:
            warnings.warn(
                f"Moving tensor from {tensor.device} to {target_device}. "
                f"This may impact performance."
            )
        tensor = tensor.to(target_device)

    return tensor


def batch_to_device(batch: Union[dict, list, tuple, 'torch.Tensor'],
                    device: Union[str, 'torch.device']) -> Union[dict, list, tuple, 'torch.Tensor']:
    """
    Recursively move all tensors in a nested structure to the specified device.

    Args:
        batch: Nested structure containing tensors (dict, list, tuple, or tensor)
        device: Target device

    Returns:
        Same structure with all tensors moved to device

    Example:
        >>> batch = {'states': torch.tensor([1, 2]), 'actions': torch.tensor([0, 1])}
        >>> batch = batch_to_device(batch, 'cpu')
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(item, device) for item in batch]
    elif isinstance(batch, tuple):
        return tuple(batch_to_device(item, device) for item in batch)
    else:
        return batch


def validate_same_device(*tensors: 'torch.Tensor') -> bool:
    """
    Check if all tensors are on the same device.

    Args:
        *tensors: Variable number of tensors to check

    Returns:
        True if all tensors are on same device

    Raises:
        ValueError: If tensors are on different devices

    Example:
        >>> t1 = torch.tensor([1, 2])
        >>> t2 = torch.tensor([3, 4])
        >>> validate_same_device(t1, t2)
        True
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")

    if len(tensors) == 0:
        return True

    # Filter out non-tensors
    tensor_list = [t for t in tensors if isinstance(t, torch.Tensor)]

    if len(tensor_list) == 0:
        return True

    first_device = tensor_list[0].device

    for i, tensor in enumerate(tensor_list[1:], start=1):
        if tensor.device != first_device:
            raise ValueError(
                f"Device mismatch: tensor 0 is on {first_device}, "
                f"but tensor {i} is on {tensor.device}"
            )

    return True


def to_hashable(state: Union['torch.Tensor', np.ndarray, List],
                decimals: int = 6) -> tuple:
    """
    Convert state (tensor, array, or list) to a hashable tuple for use as dict key.

    Args:
        state: State representation
        decimals: Number of decimal places to round to

    Returns:
        Hashable tuple

    Example:
        >>> state = torch.tensor([[1.234567, 2.345678]])
        >>> key = to_hashable(state)
        >>> isinstance(key, tuple)
        True
    """
    # Convert to numpy first
    arr = to_numpy(state)

    # Flatten and round
    flat = arr.flatten()
    rounded = np.round(flat, decimals)

    return tuple(rounded)


def safe_concatenate(*arrays: Union['torch.Tensor', np.ndarray],
                     axis: int = 0,
                     as_numpy: bool = True) -> Union['torch.Tensor', np.ndarray]:
    """
    Safely concatenate arrays/tensors, handling mixed types.

    Args:
        *arrays: Variable number of arrays or tensors
        axis: Concatenation axis
        as_numpy: If True, return numpy array; otherwise return tensor

    Returns:
        Concatenated array or tensor

    Example:
        >>> a = np.array([[1, 2]])
        >>> b = torch.tensor([[3, 4]])
        >>> result = safe_concatenate(a, b)
        >>> result.shape
        (2, 2)
    """
    if len(arrays) == 0:
        raise ValueError("Need at least one array to concatenate")

    if as_numpy:
        # Convert all to numpy
        np_arrays = [to_numpy(arr) for arr in arrays]
        return np.concatenate(np_arrays, axis=axis)
    else:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
        # Convert all to tensors
        tensors = [to_tensor(arr) if not isinstance(arr, torch.Tensor) else arr
                   for arr in arrays]
        # Ensure same device
        device = tensors[0].device
        tensors = [t.to(device) for t in tensors]
        return torch.cat(tensors, dim=axis)


def check_nan_inf(data: Union['torch.Tensor', np.ndarray],
                  name: str = "data",
                  raise_error: bool = True) -> bool:
    """
    Check for NaN or Inf values in tensor/array.

    Args:
        data: Tensor or array to check
        name: Name for error message
        raise_error: If True, raise error on NaN/Inf; otherwise return False

    Returns:
        True if data is valid (no NaN/Inf)

    Example:
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> check_nan_inf(data)
        True
    """
    arr = to_numpy(data)

    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()

    if has_nan or has_inf:
        msg = f"{name} contains "
        if has_nan:
            msg += f"NaN values ({np.isnan(arr).sum()} total)"
        if has_inf:
            if has_nan:
                msg += " and "
            msg += f"Inf values ({np.isinf(arr).sum()} total)"

        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
            return False

    return True
