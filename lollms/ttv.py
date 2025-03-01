from abc import ABC, abstractmethod
from typing import List, Optional

class LollmsTTV(ABC):
    """
    Abstract base class for text-to-video generation services.
    Subclasses must implement the methods to generate videos from text prompts.
    """
    def __init__(self, service_name):
        self.name = service_name

    @abstractmethod
    def generate_video(self, prompt: str, negative_prompt: str, num_frames: int = 49, fps: int = 8, 
                       num_inference_steps: int = 50, guidance_scale: float = 6.0, 
                       seed: Optional[int] = None) -> str:
        """
        Generates a video from a single text prompt.

        Args:
            prompt (str): The text prompt describing the video.
            negative_prompt (str): Text describing elements to avoid in the video.
            num_frames (int): Number of frames in the video. Default is 49.
            fps (int): Frames per second. Default is 8.
            num_inference_steps (int): Number of steps for the model to infer. Default is 50.
            guidance_scale (float): Controls how closely the model adheres to the prompt. Default is 6.0.
            seed (Optional[int]): Random seed for reproducibility. Default is None.

        Returns:
            str: The path to the generated video.
        """
        pass

    @abstractmethod
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
