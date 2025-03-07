from pathlib import Path
from typing import List, Optional, Dict, Any
from lollms.ttv import LollmsTTV
from lollms.app import LollmsApplication
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
from lollms.utilities import find_next_available_filename
import requests
import json
import os
import time

class LollmsNovitaAITextToVideo(LollmsTTV):
    """
    A binding for the Novita.ai Text-to-Video API.
    This class allows generating videos from text prompts using the Novita.ai service.
    """
    def __init__(
                    self,
                    app:LollmsApplication,
                    output_folder:str|Path=None
                ):
        """
        Initializes the NovitaAITextToVideo binding.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL for the Novita.ai API. Defaults to "https://api.novita.ai/v3/async".
        """
        # Check for the NOVITA_AI_KEY environment variable if no API key is provided
        api_key = os.getenv("NOVITA_AI_KEY","")
        service_config = TypedConfig(
            ConfigTemplate([
                {"name":"api_key", "type":"str", "value":api_key, "help":"A valid Novita AI key to generate text using anthropic api"},
                {"name":"model_name","type":"str","value":"darkSushiMixMix_225D_64380.safetensors", "options": ["darkSushiMixMix_225D_64380.safetensors"], "help":"The model name"}
            ]),
            BaseConfig(config={
                "api_key": "",     # use avx2
            })
        )

        super().__init__("novita_ai", app, service_config,output_folder)
        self.model_name = self.service_config.model_name
        self.base_url = "https://api.novita.ai/v3/async"

    def getModels(self):
        """
        Gets the list of models
        """
        url = "https://api.novita.ai/v3/model"
        headers = {
            "Content-Type": "<content-type>",
            "Authorization": "<authorization>"
        }

        response = requests.request("GET", url, headers=headers)
        return response.json()["models"]
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_name: str = "",
        height: int = 512,
        width: int = 512,
        steps: int = 20,
        seed: int = -1,
        nb_frames: int = 64,
        guidance_scale: Optional[float] = None,
        loras: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[Dict[str, Any]]] = None,
        closed_loop: Optional[bool] = None,
        clip_skip: Optional[int] = None,
        output_dir:str | Path =None,
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
        if model_name=="":
            model_name = self.model_name
        if output_dir is None:
            output_dir = self.output_folder
        url = f"{self.base_url}/txt2video"
        headers = {
            "Authorization": f"Bearer {self.service_config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "extra": {
                "response_video_type": "mp4", # gif
                "enterprise_plan": {"enabled": False}
            },
            "model_name": model_name,
            "height": height,
            "width": width,
            "steps": steps,
            "prompts": [
                {
                    "frames": nb_frames,
                    "prompt": prompt
                }
            ],
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "loras": loras,
            "embeddings": embeddings,
            "closed_loop": closed_loop,
            "clip_skip": clip_skip
        }  
        # Remove None values from the payload to avoid sending null fields
        payload = {k: v for k, v in payload.items() if v is not None}

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for HTTP errors
        task_id = response.json().get("task_id")


        url = f"https://api.novita.ai/v3/async/task-result?task_id={task_id}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.service_config.api_key}",
        }
        done = False
        while not done:
            response = requests.request("GET", url, headers=headers)
            infos = response.json()
            if infos["task"]["status"]=="TASK_STATUS_SUCCEED" or infos["task"]["status"]=="TASK_STATUS_FAILED":
                done = True
            time.sleep(1)
        if infos["task"]["status"]=="TASK_STATUS_SUCCEED":
            if output_dir:
                output_dir = Path(output_dir)
                file_name = output_dir/find_next_available_filename(output_dir, "vid_novita_")  # You can change the filename if needed
                self.download_video(infos["videos"][0]["video_url"], file_name )
                return file_name
        return None

    def settings_updated(self):
        pass

    def generate_video_by_frames(self, prompts: List[str], frames: List[int], negative_prompt: str, fps: int = 8, 
                       num_inference_steps: int = 50, guidance_scale: float = 6.0, 
                       seed: Optional[int] = None) -> str:
        """
        Generates a video from a list of prompts and corresponding frames.

        Args:
            prompts (List[str]): List of text prompts for each frame.
            frames (List[int]): List of frame indices corresponding to each prompt.
            negative_prompt (str): Text describing elements to avoid in the video.
            fps (int): Frames per second. Default is 8.
            num_inference_steps (int): Number of steps for the model to infer. Default is 50.
            guidance_scale (float): Controls how closely the model adheres to the prompt. Default is 6.0.
            seed (Optional[int]): Random seed for reproducibility. Default is None.

        Returns:
            str: The path to the generated video.
        """
        pass

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
            "Authorization": f"Bearer {self.service_config.api_key}",
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
