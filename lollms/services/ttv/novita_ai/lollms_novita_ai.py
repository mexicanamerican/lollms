from pathlib import Path
from typing import List, Optional, Dict, Any
from lollms.ttv import LollmsTTV
import requests
import json
import os

class LollmsNovitaAITextToVideo(LollmsTTV):
    """
    A binding for the Novita.ai Text-to-Video API.
    This class allows generating videos from text prompts using the Novita.ai service.
    """
    def __init__(self, api_key: str, base_url: str = "https://api.novita.ai/v3/async"):
        """
        Initializes the NovitaAITextToVideo binding.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL for the Novita.ai API. Defaults to "https://api.novita.ai/v3/async".
        """
        super().__init__("novita_ai")
        if api_key is None:
            # Check for the NOVITA_AI_KEY environment variable if no API key is provided
            api_key = os.getenv("NOVITA_AI_KEY","")
            if api_key is None:
                raise ValueError("No API key provided and NOVITA_AI_KEY environment variable is not set.")        
        self.api_key = api_key
        self.base_url = base_url

    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_name: str = "darkSushiMixMix_225D_64380.safetensors",
        height: int = 512,
        width: int = 512,
        steps: int = 20,
        seed: int = -1,
        guidance_scale: Optional[float] = None,
        loras: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[Dict[str, Any]]] = None,
        closed_loop: Optional[bool] = None,
        clip_skip: Optional[int] = None,
    ) -> str:
        """
        Generates a video from text prompts using the Novita.ai API.

        Args:
            model_name (str): Name of the model checkpoint.
            height (int): Height of the video, range [256, 1024].
            width (int): Width of the video, range [256, 1024].
            steps (int): Number of denoising steps, range [1, 50].
            prompts (List[Dict[str, Any]]): List of prompts with frames and text descriptions.
            negative_prompt (Optional[str]): Text input to avoid in the video. Defaults to None.
            seed (int): Random seed for reproducibility. Defaults to -1.
            guidance_scale (Optional[float]): Controls adherence to the prompt. Defaults to None.
            loras (Optional[List[Dict[str, Any]]]): List of LoRA parameters. Defaults to None.
            embeddings (Optional[List[Dict[str, Any]]]): List of embeddings. Defaults to None.
            closed_loop (Optional[bool]): Controls animation loop behavior. Defaults to None.
            clip_skip (Optional[int]): Number of layers to skip during optimization. Defaults to None.

        Returns:
            str: The task_id for retrieving the generated video.
        """
        url = f"{self.base_url}/txt2video"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model_name": model_name,
            "height": height,
            "width": width,
            "steps": steps,
            "prompts": [prompt],
            "negative_prompt": negative_prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "loras": loras,
            "embeddings": embeddings,
            "closed_loop": closed_loop,
            "clip_skip": clip_skip,
        }
        # Remove None values from the payload to avoid sending null fields
        payload = {k: v for k, v in payload.items() if v is not None}

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for HTTP errors

        return response.json().get("task_id")

    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        Retrieves the result of a video generation task using the task_id.

        Args:
            task_id (str): The task_id returned by the generate_video method.

        Returns:
            Dict[str, Any]: The task result containing the video URL and other details.
        """
        url = f"{self.base_url}/task-result"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        params = {
            "task_id": task_id,
        }

                response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors

        return response.json()

    def download_video(self, video_url: str, save_path: Path) -> None:
        """
        Downloads the generated video from the provided URL and saves it to the specified path.

        Args:
            video_url (str): The URL of the video to download.
            save_path (Path): The path where the video will be saved.
        """
        response = requests.get(video_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(save_path, "wb") as file:
            file.write(response.content)
