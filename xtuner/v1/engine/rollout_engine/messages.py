import asyncio
import time
from enum import Enum, auto
from itertools import count
from typing import Any, Dict, List, Literal, Optional, Union, cast

import shortuuid
import torch
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field

from transformers import PreTrainedTokenizer
from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto.rl_data import SampleParams


class PydanticBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )


# from lmdeploy start
class GenerateRequest(PydanticBaseModel):
    """Generate request."""

    prompt: Union[str, List[Dict[str, Any]]]
    image_url: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])
    session_id: int = -1
    interactive_mode: bool = False
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])
    request_output_len: Optional[int] = Field(default=None, examples=[None])  # noqa
    top_p: float = 0.8
    top_k: int = 40
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    cancel: Optional[bool] = False  # cancel a responding request
    adapter_name: Optional[str] = Field(default=None, examples=[None])
    seed: Optional[int] = None
    min_new_tokens: Optional[int] = Field(default=None, examples=[None])
    min_p: float = 0.0


class ModelPermission(PydanticBaseModel):
    """Model permissions."""

    id: str = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(PydanticBaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "xtuner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []


class ModelList(PydanticBaseModel):
    """Model list consists of model cards."""

    object: str = "list"
    data: List[ModelCard] = []


class Function(PydanticBaseModel):
    """Function descriptions."""

    description: Optional[str] = Field(default=None, examples=[None])
    name: str
    parameters: Optional[object] = None


class Tool(PydanticBaseModel):
    """Function wrapper."""

    type: str = Field(default="function", examples=["function"])
    function: Function


class ToolChoice(PydanticBaseModel):
    """The tool choice definition."""

    function: str
    type: Literal["function"] = Field(default="function", examples=["function"])


class JsonSchema(PydanticBaseModel):
    name: str
    # description is not used since it depends on model
    description: Optional[str] = None
    # `schema` is a reserved field in Pydantic BaseModel
    # use alias since pydantic does not support the OpenAI key `schema`
    json_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema", examples=[None])
    # strict is not used
    strict: Optional[bool] = False
    model_config = ConfigDict(serialize_by_alias=True)


class ResponseFormat(PydanticBaseModel):
    # regex_schema is extended by lmdeploy to support regex output
    type: Literal["text", "json_object", "json_schema", "regex_schema"]
    json_schema: Optional[JsonSchema] = None
    regex_schema: Optional[str] = None


class ChatCompletionRequest(PydanticBaseModel):
    """Chat completion request."""

    model: str
    messages: Union[str, List[Dict[str, Any]]] = Field(examples=[[{"role": "user", "content": "hi"}]])
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = Field(default="auto", examples=["none"])
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    n: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = Field(default=None, examples=[None])
    max_completion_tokens: Optional[int] = Field(
        default=None,
        examples=[None],
        description=(
            "An upper bound for the number of tokens that can be generated for a completion, "
            "including visible output tokens and reasoning tokens"
        ),
    )
    max_tokens: Optional[int] = Field(
        default=None,
        examples=[None],
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
    )
    stop: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])

    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    response_format: Optional[ResponseFormat] = Field(default=None, examples=[None])
    # additional argument of lmdeploy
    do_preprocess: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.0
    session_id: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    top_k: Optional[int] = 40
    seed: Optional[int] = None
    min_new_tokens: Optional[int] = Field(default=None, examples=[None])
    min_p: float = 0.0
    enable_thinking: Optional[bool] = None
    return_token_ids: Optional[bool] = False
    include_stop_str_in_output: Optional[bool] = False

    def model_post_init(self, __context: Any) -> None:
        if not self.max_completion_tokens:
            self.max_completion_tokens = self.max_tokens


# from lmdeploy end


class ResultEvent:
    def __init__(self):
        self.event = asyncio.Event()
        self.result = None

    def set_result(self, result):
        self.result = result
        self.event.set()

    async def wait_for_result(self):
        await self.event.wait()
        return self.result


class MessagesHolder:
    def __init__(self, chat_template: Template, tokenizer: PreTrainedTokenizer):
        # this class is used to manage the messages and input_ids, to make sure new tokens are appended correctly
        self.messages: list[dict[str, str]] = []
        self.chat_template = chat_template
        self.tokenizer = tokenizer

        self.input_ids: list[int] = []
        self.text = ""
        self.seen_tokens = 0

    def decode_tokens(self, tokens: list[int], reason="stop"):
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        if reason == "stop":
            stop_token = self.tokenizer.decode(tokens[-1:], skip_special_tokens=False)
            output_text = text[: -len(stop_token)]
        else:
            output_text = text
        self.text += text
        self.input_ids.extend(tokens)
        self.messages.append({"role": "assistant", "content": output_text})
        self.seen_tokens += len(tokens)
        return output_text

    def encode_messages(self, messages: list[dict[str, str]]) -> torch.Tensor:
        self.messages.extend(messages)
        text = self.get_text()

        assert text.startswith(self.text), f"Template Error: {text} does not start with {self.text}"

        append_text = text[len(self.text) :]
        self.text += append_text
        tokens = cast(torch.Tensor, self.tokenizer(append_text, return_tensors="pt").input_ids)
        self.seen_tokens += len(tokens)
        return tokens

    def get_text(self, add_generation_prompt=True):
        return self.chat_template.render(messages=self.messages, add_generation_prompt=add_generation_prompt)


class SessionItem:
    def __init__(self, resource_id: int, msg_holder: MessagesHolder):
        self.resource_id = resource_id
        self.msg_holder = msg_holder
        self.gen_config = GenerateConfig()


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: torch.Tensor, sampling_params: SampleParams):
        """初始化序列对象.

        Args:
            token_ids: torch.LongTensor 类型的输入token，支持1D或2D张量
            sampling_params: SamplingParams 对象，包含采样参数
        """
        self.token_ids: torch.Tensor
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING

        # 确保 token_ids 是 torch.LongTensor 类型
        if not isinstance(token_ids, torch.LongTensor):
            if isinstance(token_ids, torch.Tensor):
                self.token_ids = token_ids.long()
            else:
                # 如果传入的是list或其他类型，转换为LongTensor
                self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        else:
            self.token_ids = token_ids.clone()

        # 如果是2D张量，确保只有一个序列
        if self.token_ids.dim() == 2:
            assert self.token_ids.size(0) == 1, "Only support single sequence in batch dimension"
            self.token_ids = self.token_ids.squeeze(0)
        elif self.token_ids.dim() == 0:
            # 如果是标量，转换为1D张量
            self.token_ids = self.token_ids.unsqueeze(0)
        elif self.token_ids.dim() > 2:
            raise ValueError("token_ids should be 1D or 2D tensor")

        self.last_token = self.token_ids[-1].item()
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """返回提示部分的token IDs."""
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """返回生成部分的token IDs."""
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取第i个block的tokens."""
        assert 0 <= i < self.num_blocks
        start_idx = i * self.block_size
        end_idx = min((i + 1) * self.block_size, self.num_tokens)
        return self.token_ids[start_idx:end_idx]

    def append_token(self, token_id: int):
        """追加新的token到序列末尾."""
        new_token = torch.tensor([token_id], dtype=torch.long, device=self.token_ids.device)
        self.token_ids = torch.cat([self.token_ids, new_token])
        self.last_token = token_id
        self.num_tokens += 1

    def get_tokens_as_list(self):
        """返回tokens的Python list格式."""
        return self.token_ids.tolist()

    def allocate_blocks(self, block_ids: list[int]):
        """分配block table给当前序列."""
        required_blocks = self.num_blocks
        if len(block_ids) < required_blocks:
            raise ValueError(f"Insufficient blocks: need {required_blocks}, got {len(block_ids)}")
        self.block_table = block_ids[:required_blocks]

    def extend_block_table(self, additional_blocks: list[int]):
        """扩展block table以支持更多tokens."""
        self.block_table.extend(additional_blocks)

    def set_status(self, status: SequenceStatus):
        """设置序列状态."""
        self.status = status

    def mark_finished(self):
        """标记序列为已完成."""
        self.status = SequenceStatus.FINISHED

    def mark_running(self):
        """标记序列为运行中."""
        self.status = SequenceStatus.RUNNING

    def __getstate__(self):
        """序列化状态，用于保存/恢复."""
        return (
            self.seq_id,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
            self.temperature,
            self.max_tokens,
            self.status,
        )

    def __setstate__(self, state):
        """反序列化状态，用于保存/恢复."""
        (
            self.seq_id,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            token_data,
            self.temperature,
            self.max_tokens,
            self.status,
        ) = state

        if self.num_completion_tokens == 0:
            self.token_ids = token_data
        else:
            self.last_token = token_data
            # 需要重建完整的token_ids（这里可能需要额外的信息）
            # 在实际使用中，可能需要其他方式来恢复完整序列
            self.token_ids = torch.tensor([self.last_token], dtype=torch.long)
