import asyncio
import time
import uuid
from queue import Queue
from typing import Dict, cast

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from jinja2 import Template

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.utils import get_logger

from .messages import (
    ChatCompletionRequest,
    MessagesHolder,
    ModelCard,
    ModelList,
    ResultEvent,
    SessionItem,
)


logger = get_logger()


class TaskItem:
    priority = -1

    def __init__(self, resource_id: int):
        self.resource_id = resource_id

    def __lt__(self, other: "TaskItem"):
        return self.priority < other.priority


class InferItem(TaskItem):
    priority = 1

    def __init__(self, resource_id: int, input_ids: torch.Tensor, context_length: int, sample_params: SampleParams):
        super().__init__(resource_id)
        self.sample_params = sample_params
        self.input_ids = cast(torch.LongTensor, input_ids.reshape(1, -1))
        self.raw_input_ids = self.input_ids.clone()
        self.generated_tokens: list[int] = []
        self.first_token = True
        self.context_length = context_length

    def update_with_token(self, new_token: torch.Tensor):
        if not self.first_token:
            input_ids_int = cast(int, self.input_ids.item())
            self.generated_tokens.append(input_ids_int)
        self.input_ids = cast(torch.LongTensor, new_token)
        self.first_token = False

    def should_stop(self):
        if (
            (len(self.generated_tokens) >= 1 and self.generated_tokens[-1] in self.sample_params.stop_token_ids)
            or (self.raw_input_ids.numel() + len(self.generated_tokens)) >= self.context_length
            or len(self.generated_tokens) >= self.sample_params.max_tokens
        ):
            return True
        else:
            return False


class ResetItem(TaskItem):
    priority = 0


class ResultItem(TaskItem):
    def __init__(self, resource_id, result):
        super().__init__(resource_id)
        self.result = result


class APIServer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: Template | None = None,
        queue: Queue[TaskItem] | None = None,
        result_queue: Queue[ResultItem] | None = None,
        url: str = "0.0.0.0",
        port: int = 10240,
        model_name: str = "",
        resource_pool: list[int] | None = None,
        context_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template if chat_template else Template(tokenizer.get_chat_template())

        self.queue = queue or Queue()
        self.result_queue = result_queue or Queue()

        self.url = url
        self.port = port
        self.task = None
        self.server = None
        self.model_name = model_name
        self.resource_pool: asyncio.Queue[int] = asyncio.Queue()
        for r in resource_pool or []:
            self.resource_pool.put_nowait(r)
        self.session_map: Dict[int, SessionItem] = {}
        self.context_length = context_length
        self.event_map: Dict[int, ResultEvent] = {}
        self.inspect_result_task = None

        self.launched_time = time.time()
        self.first_request = True

        self.default_stop_tokens = [self.tokenizer.eos_token_id]
        logger.info(f"Use eos token {self.tokenizer.eos_token}({self.tokenizer.eos_token_id}) as default stop token")

    # tasks
    async def acquire_session(self, session_id: int) -> SessionItem:
        if session_id in self.session_map:
            return self.session_map[session_id]
        else:
            resource = await self.resource_pool.get()
            self.session_map[session_id] = SessionItem(resource, MessagesHolder(self.chat_template, self.tokenizer))
            return self.session_map[session_id]

    async def release_session(self, session_id: int):
        session = self.session_map.pop(session_id, None)
        if session is not None:
            await self.run_generate(ResetItem(session.resource_id))
            await self.resource_pool.put(session.resource_id)

    async def execute_task(self, session_id: int, messages: list[dict[str, str]], sample_params: SampleParams):
        session = await self.acquire_session(session_id)

        # update gen config
        if sample_params.stops and not sample_params.stop_token_ids:
            stop_tokens = [self.tokenizer.encode(token)[0] for token in sample_params.stops]
            stop_tokens.extend(self.default_stop_tokens)
            sample_params.stop_token_ids = stop_tokens

        # end
        if self.context_length <= session.msg_holder.seen_tokens:
            return "", 0, 0, "length"
        else:
            input_ids = session.msg_holder.encode_messages(messages)
            infer_item = InferItem(
                session.resource_id, torch.tensor(input_ids).reshape([1, -1]), self.context_length, sample_params
            )
            generated_tokens = await self.run_generate(infer_item)
            reason = "stop" if generated_tokens[-1] in sample_params.stop_token_ids else "length"
            decode_text = session.msg_holder.decode_tokens(generated_tokens, reason)
            return (decode_text, input_ids, generated_tokens, reason)

    # apis
    async def launch(self):
        app = FastAPI(title="OpenAI-compatible API")

        app.post("/v1/chat/completions")(self.chat_completion)
        app.post("/v1/chat/interactive")(self.chat_interactive)
        app.get("/health")(self.health)
        app.get("/v1/models")(self.available_models)

        config = uvicorn.Config(app, host=self.url, port=self.port, log_level="error")
        server = uvicorn.Server(config)
        loop = asyncio.get_event_loop()
        task = loop.create_task(server.serve())

        # Wait for server to start
        while not server.started:
            await asyncio.sleep(0.001)

        self.task = task
        self.server = server

        self.inspect_result_task = asyncio.create_task(self.inspect_result())
        logger.info(f"API server launched successfully on port {self.port}")
        self.reset_launch_time()

    async def chat_completion(self, request: ChatCompletionRequest):
        if self.first_request:
            logger.info(f"First request time: {time.time() - self.launched_time:.2f} seconds")
            self.first_request = False

        sample_params = SampleParams(**request.model_dump(include=set(SampleParams.model_fields.keys())))
        session_id = uuid.uuid4().int
        if isinstance(request.messages, str):
            messages = [{"role": "user", "content": request.messages}]
        else:
            messages = request.messages
        decode_text, input_ids, gened_ids, reason = await self.execute_task(session_id, messages, sample_params)
        await self.release_session(session_id)

        return {
            "id": str(session_id),
            "object": "chat.completion",
            "created": 0,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": decode_text,
                        "output_ids": gened_ids,
                        "input_ids": input_ids,
                    },
                    "finish_reason": reason,
                }
            ],
            "usage": dict(
                completion_tokens=len(gened_ids),
                prompt_tokens=len(input_ids),
                total_tokens=len(gened_ids) + len(input_ids),
            ),
        }

    # apis for server status
    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)

    async def available_models(self):
        """Show available models."""
        return ModelList(data=[ModelCard(id=self.model_name, root=self.model_name)])

    # stop server
    async def stop_server(self):
        if self.task is not None:
            self.server.should_exit = True
            await self.task
            self.task = None
            self.server = None
            self.inspect_result_task.cancel()
            self.inspect_result_task = None
            logger.info(f"API server on {self.port} stopped successfully")

    async def wait_closed(self):
        if self.task is not None:
            await self.task
            self.task = None
            self.server = None
        else:
            print("Server is not running, nothing to wait for.")

    # protocol for infer
    async def run_generate(self, item: TaskItem):
        event = ResultEvent()
        self.event_map[item.resource_id] = event
        self.queue.put(item)
        await asyncio.sleep(0)
        generated_tokens = await event.wait_for_result()
        self.event_map.pop(item.resource_id, None)
        return generated_tokens

    async def inspect_result(self):
        while True:
            if self.result_queue.empty():
                pass
            else:
                item: ResultItem = self.result_queue.get()
                event = self.event_map.get(item.resource_id)
                event.set_result(item.result)
            await asyncio.sleep(0)

    def reset_launch_time(self):
        self.launched_time = time.time()
        self.first_request = True

    def change_max_length(self, max_length: int):
        self.context_length = max_length
