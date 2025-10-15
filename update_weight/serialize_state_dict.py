from typing import Dict

import torch
from mp_serializer import MultiprocessingSerializer
from tensor_bucket import FlattenedTensorBucket
from patch_torch import monkey_patch_torch_reductions


def state_dict_to_serialized_flattened_tensor(
    state_dict: Dict[str, torch.Tensor],
        output_str: bool = True,
    ) -> str | bytes:
    """
    Convert a state_dict to a serialized flattened tensor format.

    Args:
        state_dict (Dict[str, torch.Tensor]): The state dictionary to convert.
        output_str (bool): If True, return a base64-encoded string; otherwise, return raw bytes.

    Returns:
        str or bytes: The serialized flattened tensor.
    """
    monkey_patch_torch_reductions()

    named_tensors = list(state_dict.items())
    if len(named_tensors) == 0:
        output = "" if output_str else b""
        return output

    bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    flattened_tensor_bucket_dict = dict(
        metadata=bucket.get_metadata(),
        flattened_tensor=bucket.get_flattened_tensor(),
    )
    output = MultiprocessingSerializer.serialize(flattened_tensor_bucket_dict, output_str=output_str)

    return output


def serialized_flattened_tensor_to_state_dict(
    serialized_data: str | bytes
) -> Dict[str, torch.Tensor]:
    """
    Convert a serialized flattened tensor format back to a state_dict.

    Args:
        serialized (str | bytes): The serialized flattened tensor.

    Returns:
        Dict[str, torch.Tensor]: The reconstructed state dictionary.
    """
    monkey_patch_torch_reductions()

    if not serialized_data:
        return dict()

    flattened_tensor_bucket_dict = MultiprocessingSerializer.deserialize(serialized_data)
    assert "metadata" in flattened_tensor_bucket_dict and "flattened_tensor" in flattened_tensor_bucket_dict

    bucket = FlattenedTensorBucket(**flattened_tensor_bucket_dict)
    reconstructed_tensors = bucket.reconstruct_tensors()

    return dict(reconstructed_tensors)
