import copy

import torch

from xtuner.v1.model.base import BaseModel
from xtuner.v1.utils import get_logger


logger = get_logger()


class BaseEngine:
    def __init__(
        self,
        model: BaseModel,
        batch_size=128,
        max_length=4096,
        max_prefill_length=4096,
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prefill_length = max_prefill_length

    # interface

    @torch.inference_mode()
    def prefill(self, batch_input_ids: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        """
        batch_input_ids: dict['batch_id':input_ids]
        """
        raise NotImplementedError("This method should be implemented in subclasses")

    @torch.inference_mode()
    def decode(self, batch_input_ids: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        """
        batch_input_ids: dict['batch_id':input_ids]
        """
        raise NotImplementedError("This method should be implemented in subclasses")

    @torch.inference_mode()
    def reset(self, batch_idx):
        raise NotImplementedError("This method should be implemented in subclasses")

    @torch.inference_mode()
    def prepare_for_inference(self):
        pass

    @torch.inference_mode()
    def end_inference(self):
        pass

    # for forward

    @torch.inference_mode()
    def iter_prefill(self, batch_input_ids: list[tuple[int, torch.Tensor]]):
        device = batch_input_ids[0][1].device

        cur_length = 0
        cur_batch_input_ids: list[tuple[int, torch.Tensor]] = []
        decode_token = []

        while len(batch_input_ids) > 0:
            batch_id, input_ids = batch_input_ids.pop(0)
            if cur_length + input_ids.numel() <= self.max_prefill_length:
                cur_batch_input_ids.append((batch_id, input_ids))
                cur_length += input_ids.numel()
                decode_token.append(True)
            else:
                used_length = self.max_prefill_length - cur_length
                cur_batch_input_ids.append((batch_id, input_ids[:, :used_length]))
                cur_length += used_length
                decode_token.append(False)
                batch_input_ids.insert(0, (batch_id, input_ids[:, used_length:]))
            if cur_length == self.max_prefill_length:
                yield cur_batch_input_ids, torch.tensor(decode_token, device=device, dtype=torch.bool)
                cur_batch_input_ids = []
                decode_token = []
                cur_length = 0
        if len(cur_batch_input_ids) > 0:
            yield cur_batch_input_ids, torch.tensor(decode_token, device=device, dtype=torch.bool)

    @torch.inference_mode()
    def prefill_wrapper(self, batch_input_ids: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        """Wrapper for prefill method to return logits for each batch_id."""
        try:
            outputs = []
            for part_batch_input_ids, mask in self.iter_prefill(copy.copy(batch_input_ids)):
                output = self.prefill(part_batch_input_ids)
                assert isinstance(output, torch.Tensor)
                outputs.append(output[mask])
            return torch.cat(outputs, dim=0)
        except Exception as e:
            logger.error(f"Error in prefill: {e}")
            raise e

    @torch.inference_mode()
    def decode_wrapper(self, batch_input_ids: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        """Wrapper for decode method to return logits for each batch_id."""
        try:
            output = self.decode(batch_input_ids)
            assert isinstance(output, torch.Tensor)
            return output
        except Exception as e:
            logger.error(f"Error in decode: {e}")
            raise e

    def change_max_length(self, max_length):
        # total_memory = self.batch_size * self.max_length
        self.max_length = max_length
        # self.batch_size = total_memory // self.max_length
        logger.info(f"[Engine] adjust max length to {self.max_length} with {self.batch_size} bs")
        return self.batch_size
