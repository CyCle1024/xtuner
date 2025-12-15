from collections import deque

import torch

from xtuner.v1.data_proto.rl_data import SampleParams

from .block_manager import BlockManager
from .messages import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int, eos: int, dp_size: int, min_num_seqs: int = 1):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos
        self.dp_size = dp_size
        self.min_num_seqs = min_num_seqs
        self.waiting: list[deque[Sequence]] = [deque() for _ in range(dp_size)]
        self.running: list[deque[Sequence]] = [deque() for _ in range(dp_size)]
        self.dp_routing_record: dict[Sequence, int] = dict()

    def _requests_count(self, dp_rank: int) -> int:
        return len(self.waiting[dp_rank]) + len(self.running[dp_rank])

    def _get_dummy_requests(self, ref_seq: Sequence, block_manager: BlockManager) -> Sequence:
        """It requires at least two requests in micro batch decoding function.

        When there is only one request, we need a dummy request to build the batch. ref_seq is the sequence for
        reference.
        """
        dummy_token_ids = torch.tensor([[0]], dtype=torch.long, device=ref_seq.token_ids.device)
        ref_sampling_params = SampleParams(temperature=ref_seq.temperature, max_tokens=ref_seq.max_tokens)
        dummy_seq = Sequence(dummy_token_ids, sampling_params=ref_sampling_params)
        dummy_seq.status = ref_seq.status

        if block_manager.can_allocate(dummy_seq):
            block_manager.allocate(dummy_seq)
        else:
            raise RuntimeError("Cannot allocate dummy sequence in block manager.")
        return dummy_seq

    def is_finished(self):
        return all(not q for q in self.waiting) and all(not q for q in self.running)

    def add(self, seq: Sequence):
        # add seq to the dp_rank with the least requests
        num_requests_all_dp = [self._requests_count(dp_rank) for dp_rank in range(self.dp_size)]
        dp_rank = num_requests_all_dp.index(min(num_requests_all_dp))
        self.dp_routing_record[seq] = dp_rank
        self.waiting[dp_rank].append(seq)

    def schedule(self, block_manager_list: list[BlockManager]) -> tuple[list[list[Sequence]], bool]:
        scheduled_seqs: list[list[Sequence]] = [[]] * self.dp_size
        num_seqs = [0] * self.dp_size
        num_batched_tokens = [0] * self.dp_size
        # prefill
        while any(
            can_schedule_prefill := [
                len(self.waiting[dp_rank]) > 0 and num_seqs[dp_rank] < self.max_num_seqs
                for dp_rank in range(self.dp_size)
            ]
        ):
            for dp_rank, dp_waiting in enumerate(self.waiting):
                if not can_schedule_prefill[dp_rank]:
                    continue
                seq = dp_waiting[0]
                if num_batched_tokens[dp_rank] + len(seq) > self.max_num_batched_tokens or not block_manager_list[
                    dp_rank
                ].can_allocate(seq):
                    continue
                num_seqs[dp_rank] += 1
                block_manager_list[dp_rank].allocate(seq)
                num_batched_tokens[dp_rank] += len(seq) - seq.num_cached_tokens
                seq.status = SequenceStatus.RUNNING
                dp_waiting.popleft()
                self.running[dp_rank].append(seq)
                scheduled_seqs[dp_rank].append(seq)
        if all(scheduled_seqs):
            return scheduled_seqs, True
        elif any(scheduled_seqs):
            # some dp ranks have scheduled sequences, others have not
            # TODO(chenchiyu): fill dummy or previous prompt if some dp_rank has no waiting sequences
            raise RuntimeError("Unsupported partial scheduling in prefill phase.")

        # decode
        while any(
            can_schedule_decode := [
                len(self.running[dp_rank]) > 0 and num_seqs[dp_rank] < self.max_num_seqs
                for dp_rank in range(self.dp_size)
            ]
        ):
            for dp_rank, dp_running in enumerate(self.running):
                if not can_schedule_decode[dp_rank]:
                    continue
                seq = dp_running.popleft()
                while not block_manager_list[dp_rank].can_append(seq):
                    if dp_running:
                        self.preempt(dp_running.pop(), block_manager_list)
                    else:
                        self.preempt(seq, block_manager_list)
                        break
                else:
                    num_seqs[dp_rank] += 1
                    block_manager_list[dp_rank].may_append(seq)
                    scheduled_seqs[dp_rank].append(seq)
        assert any(scheduled_seqs), "At least one dp rank should have scheduled sequences."

        for dp_rank, scheduled_seqs_per_dp in enumerate(scheduled_seqs):
            self.running[dp_rank].extendleft(reversed(scheduled_seqs_per_dp))
            if self.min_num_seqs > len(scheduled_seqs_per_dp):
                num_dummies = self.min_num_seqs - len(scheduled_seqs_per_dp)
                for _ in range(num_dummies):
                    dummy_seq = self._get_dummy_requests(scheduled_seqs_per_dp[0], block_manager_list[dp_rank])
                scheduled_seqs_per_dp.append(dummy_seq)
        return scheduled_seqs, False

    def preempt(self, seq: Sequence, block_manager_list: list[BlockManager]):
        dp_rank = self.dp_routing_record[seq]
        seq.status = SequenceStatus.WAITING
        block_manager_list[dp_rank].deallocate(seq)
        self.waiting[dp_rank].appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], block_manager_list: list[BlockManager]):
        for seq, token_id in zip(seqs, token_ids):
            dp_rank = self.dp_routing_record[seq]
            block_manager = block_manager_list[dp_rank]
            # Skip processing dummy sequences (identified by single zero token)
            if len(seq.token_ids) == 1 and seq.token_ids[0].item() == 0:
                # Clean up dummy sequence resources
                block_manager.deallocate(seq)
                continue

            seq.append_token(token_id)
            if token_id == self.eos or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                block_manager.deallocate(seq)
                self.running[dp_rank].remove(seq)
