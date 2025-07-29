
import aiohttp
import asyncio
import random
import io
from PIL import Image
from typing import Optional, Dict, Any
import base64
import os
import sys
import re
import logging

_logger = logging.getLogger(__name__)

# from backend import Chat

class SwarmGenerator:
    """
    An asynchronous client for generating images with the SwarmUI API.

    This class handles session management, request retries on session
    invalidation, and provides a simple interface to generate images.

    Usage:
        async with SwarmGenerator() as generator:
            image = await generator.generate(
                positive_prompt="a beautiful photograph of a cat",
                negative_prompt="ugly, deformed"
            )
            image.save("cat_photo.png")
    """

    def __init__(self,
                 server_address: str = "http://localhost:7801",
                 model: str = "sd1.5/dreamshaper_8",
                 width: int = 512,
                 height: int = 512,
                 steps: int = 20,
                 cfg_scale: float = 5.0,
                 seed: int = -1,
                 **kwargs):
        """
        Initializes the SwarmGenerator.

        Args:
            server_address (str): The full address of the SwarmUI server.
            model (str): The model to use for generation.
            width (int): The width of the generated image.
            height (int): The height of the generated image.
            steps (int): The number of generation steps.
            cfg_scale (float): The CFG scale for generation.
            seed (int): The seed for generation. -1 for random.
        """
        self.server_address = server_address.rstrip('/')
        self.default_params = {
            "model": model,
            "width": width,
            "height": height,
            "steps": steps,
            "cfgscale": cfg_scale,
            "seed": seed,
            "images": 1  # We generate one image at a time
        }
        self.default_params.update(kwargs)
        self._session_id: Optional[str] = None
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager to initialize the client session."""
        self._aiohttp_session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager to close the client session."""
        if self._aiohttp_session:
            await self._aiohttp_session.close()

    async def _get_new_session_id(self):
        """Fetches a new session_id from the SwarmUI API."""
        if not self._aiohttp_session:
            raise RuntimeError("aiohttp session not initialized.")

        url = f"{self.server_address}/API/GetNewSession"
        _logger.debug("Requesting new SwarmUI session...")
        async with self._aiohttp_session.post(url, json={}) as response:
            response.raise_for_status()
            data = await response.json()
            if "session_id" not in data:
                raise ConnectionError("Failed to get a session_id from SwarmUI.")
            self._session_id = data["session_id"]
            _logger.debug(f"Acquired session ID: {self._session_id[:10]}...")

    async def _post_api_request(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes a POST request to a SwarmUI API route, handling session state.
        """
        if not self._aiohttp_session:
            raise RuntimeError("aiohttp session not initialized.")
        if not self._session_id:
            await self._get_new_session_id()

        # Add current session ID to the payload
        payload["session_id"] = self._session_id
        url = f"{self.server_address}/API/{route}"

        # First attempt async with self._aiohttp_session.post(url, json=payload) as response:
        async with self._aiohttp_session.post(url, json=payload) as response:
            response.raise_for_status()
            json_response = await response.json()

        # If session was invalid, get a new one and retry exactly once.
        if json_response.get("error_id") == "invalid_session_id":
            _logger.debug("Session ID invalid, refreshing and retrying...")
            await self._get_new_session_id()
            payload["session_id"] = self._session_id  # Update payload with new ID
            async with self._aiohttp_session.post(url, json=payload) as response:
                response.raise_for_status()
                json_response = await response.json()

        return json_response

    async def prompt_fill(self, prompt: str) -> str:
        """Expands a prompt, including wildcards and special syntax."""
        json_response = await self._post_api_request("TestPromptFill", {'prompt': prompt})
        return json_response['result']

    async def get_params(self, include_models: bool = False) -> list[dict]:
        """
        Returns a list of parameter description objects for all valid generation parameters.
        """
        json_response = await self._post_api_request("ListT2IParams", {})
        if not include_models:
            if "models" in json_response:
                del json_response["models"]
            if "list" in json_response:
                # Prevent list of models / loras / etc showing up
                for p in json_response["list"]:
                    if "ampler" in p["name"]: continue
                    if "value_names" in p and p["value_names"] and len(p["value_names"]) > 32:
                        print(f"deleting {p['name']} value_names")
                        p["value_names"] = None
                    if "values" in p and p["values"] and len(p["values"]) > 32:
                        print(f"deleting {p['name']} values")
                        p["values"] = None
        return json_response

    async def generate(self, positive_prompt: str, negative_prompt: str = "", **kwargs) -> Image.Image:
        """
        Generates an image using the configured parameters and provided prompts.

        Args:
            positive_prompt (str): The positive text prompt for the image.
            negative_prompt (str): The negative text prompt.

        Returns:
            PIL.Image: The generated image.

        Raises:
            Exception: If the API returns an error after handling session logic.
            RuntimeError: If the client is used outside of an 'async with' block.
        """
        # Other args of note: (TODO: Determine what is acceptable as an image here)

        # Initial image input and masking (white = change, black = no change)
        ## initimage (image), initimagecreativity (float), initimageresettonorm (float), initimagenoise (float)
        ## maskimage (image), maskshrinkgrow (int), maskblur (int), maskgrow (int)
        ## unsamplerprompt (str)

        # Refining
        ## refinercontrolpercentage (float), refinerupscale (float), refinerdotiling (bool),
        ## refinermodel (str), refinersteps (int), refinercfgscale (float)

        # Image to Video
        ## videomodel (str), videoframes (int), videosteps (int), videocfg (float),
        ## videoendimage (image), videofps (int)

        # LoRAs (NOTE: I think lists are just comma separated?)
        ## loras (list[str]), loraweights (list[float]), loratencweights (list[float])

        # Misc
        ## torchcompile ("Disabled", "inductor", "cudagraphs") (cudagraphs didn't work for me)
        ## outputintermediateimages (bool), donotsaveintermediates (bool)

        # TODO: ControlNet stuff if I need it


        generation_payload = {
            **self.default_params,
        }
        if kwargs:
            generation_payload.update(kwargs)
        generation_payload["prompt"] = positive_prompt
        if negative_prompt:
            generation_payload["negativeprompt"] = negative_prompt

        json_response = await self._post_api_request("GenerateText2Image", generation_payload)

        # Check for other errors after the request and potential retry
        if "error" in json_response:
            raise Exception(f"SwarmUI API Error: {json_response['error']}")
        if "images" not in json_response or not json_response["images"]:
            raise Exception(f"API response did not contain image data: {json_response}")

        # Download the generated image
        image_relative_url = json_response["images"][0]
        image_full_url = f"{self.server_address}/{image_relative_url.lstrip('/')}"

        _logger.info(f"Downloading image from: {image_full_url}")
        async with self._aiohttp_session.get(image_full_url) as image_response:
            image_response.raise_for_status()
            image_bytes = await image_response.read()

        return Image.open(io.BytesIO(image_bytes))



async def main():
    from rich import print
    async with SwarmGenerator() as swarmui:
        for _ in range(5):
            result = await swarmui.prompt_fill("<repeat[1-5]:<random:a,b,c>>")
            print(result)
        result = await swarmui.get_params()
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
