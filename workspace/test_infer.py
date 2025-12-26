import torch
import torch.distributed as dist
import transformers

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.model.moe.moe import BalancingLossConfig, ZLossConfig
from xtuner.v1.rl.base import WorkerConfig, TrainingWorker as BaseTrainingWorker
from xtuner.v1.rl.grpo.loss import GRPOLossConfig as LossConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.utils import ForwardState


MODEL_PATH = "/mnt/shared-storage-user/llmrazor-share/model/Qwen3-8B"

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

fsdp_config = FSDPConfig()
model_cfg = get_model_config_from_hf(model_path=MODEL_PATH)
model_cfg.compile_cfg = False
with torch.device("meta"):
    model = model_cfg.build()
model.fully_shard(fsdp_config)

# load weights!!!
model.from_hf(hf_path=MODEL_PATH, strict=True)

model._set_reshard_after_forward(True)

block_size=256
model.build_kv_cache(
    max_batch_size=2,
    max_length=256,
    block_size=block_size,
)

input_texts = ["你好，能否给一些学习Rust编程的入门介绍呢？", "接下来我们写一个Python函数来计算两个数的和："]
inputs_ids = [tokenizer(text, return_tensors="pt").input_ids for text in input_texts]
block_tables = [torch.tensor([[0]], device='cuda', dtype=torch.int32), torch.tensor([[1]], device='cuda', dtype=torch.int32)]
seq_ctx_list = [SequenceContext.from_input_ids(input_ids=[input_id], state="prefilling", block_table=block_table) 
                for input_id, block_table in zip(inputs_ids, block_tables)]

with torch.no_grad():
    cur_seq_ctx = seq_ctx_list[dist.get_rank()]
    prompt_len = cur_seq_ctx.input_ids.numel()
    history_len = prompt_len
    resp = []
    for _ in range(128):
        res = model(cur_seq_ctx, loss_ctx=None)
        logits = res["logits"]
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        cur_seq_ctx = SequenceContext.from_input_ids(
            input_ids=(next_token,),
            past_kv_lens=(history_len,),
            state=ForwardState.DECODING,
            block_table=cur_seq_ctx.block_table,
        )
        history_len += 1
        resp.append(next_token)
    print(f"Rank {dist.get_rank()} Prompt: {input_texts[dist.get_rank()]}")
    # print(f"Rank {dist.get_rank()} Generated tokens: {torch.cat(resp, dim=1)[0]}")
    print(f"Rank {dist.get_rank()} Generated: {tokenizer.decode(torch.cat(resp, dim=1)[0])}")
