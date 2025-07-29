import os
import openai
import agents
import asyncio
import logging
from typing import Callable

import util

from constants import BASE_URL, API_KEY, DEFAULT_MODEL

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_client = openai.AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
agents.set_default_openai_api("chat_completions")
agents.set_default_openai_client(client=_client, use_for_tracing=False)
agents.set_tracing_disabled(disabled=True)
_model_provider = agents.OpenAIProvider(base_url=BASE_URL, api_key=API_KEY, use_responses=False)

def get_model(model_name=None) -> agents.Model:
    return _model_provider.get_model(model_name or DEFAULT_MODEL)

class Agent(agents.Agent):
    def __init__(self, name: str, instructions: str | Callable | None = None, *args, debug:bool = False, **kwargs):
        model_name = kwargs.get('model', None) or DEFAULT_MODEL
        kwargs['model'] = get_model(model_name)
        self.debug = debug
        super().__init__(name, instructions, *args, **kwargs)

    async def run(self, input:str | list[agents.TResponseInputItem], **kwargs):
        if self.debug:
            _logger.debug(f"run('{self.model.model}|{self.name}', {input})")
        result = await agents.Runner.run(self, input, **kwargs)
        if self.debug:
            _logger.info(f"<{self.model.model}|{self.name}>\n{result.final_output}")
        return result

    async def query(self, input:str | list[agents.TResponseInputItem], **kwargs):
        """Calls `run` and returns only the final output"""
        result = await self.run(input, **kwargs)
        return result.final_output

def attach_images(message:str, *images):
    content = [{"type": "input_text", "text": message}]
    for i in images:
        b64 = util.encode_image(i)
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
    return content

class Chat:
    def __init__(self, agent:Agent, prefill:list[str]|None=None):
        self.agent = agent
        self.message_history = []
        if prefill:
            role = 'user'
            for m in prefill:
                self.message_history.append({'role': role, 'content': m})
                role = 'user' if role == 'assistant' else 'assistant'
            # Ensure the call to chat will always give alternating roles
            if self.message_history and self.message_history[-1]['role'] == 'user':
                self.message_history.append({'role': 'assistant', 'content': ''})
        self.last_text = ""
        self.last_output = ""

    async def run(self, message, prefill=None, *args, **kwargs):
        self.message_history.append({"content": message, "role": "user"})
        if prefill:
            self.message_history.append({"role": "assistant", "content": prefill})
        retries = 3
        while retries:
            try:
                result = await self.agent.run(self.message_history, *args, **kwargs)
                break
            except Exception as e:
                print(f"Exception, waiting and then retrying: {e}")
                await asyncio.sleep(1.0)
                retries -= 1
                if retries > 0:
                    continue
                raise e
        self.message_history = result.to_input_list()
        self.last_text = agents.ItemHelpers.text_message_outputs(result.new_items)
        self.last_output = result.final_output
        return result

    async def chat(self, message, prefill=None, *args, **kwargs):
        result = await self.run(message, prefill=prefill, *args, **kwargs)
        return agents.ItemHelpers.text_message_outputs(result.new_items)

    def get_history(self, include_images: bool = False):
        """Returns the chat history as a list of standard role+content
        dictionaries. Optionally includes images."""
        result = []
        for m in self.message_history:
            role = m.get('role', '')
            if role != 'user' and role != 'assistant':
                continue

            entry = {'role': role}
            content = m.get('content', '')

            if isinstance(content, str):
                entry['content'] = content
                result.append(entry)
                continue

            entry['content'] = ''

            for c in content:
                if c.get('type', '') == 'input_image':
                    if not include_images:
                        continue
                    imgs = entry.get('images', [])
                    imgs.append(c['image_url'])
                    entry['images'] = imgs

                if text := c.get('text', ''):
                    entry['content'] += text
            result.append(entry)

        return result

