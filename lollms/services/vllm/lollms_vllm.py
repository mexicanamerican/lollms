# Title vLLM service
# Licence: MIT
# Author : Paris Neo
# This is a service launcher for the vllm server by Jeffrey Morgan (jmorganca)
# check it out : https://github.com/jmorganca/vllm
# Here is a copy of the LICENCE https://github.com/jmorganca/vllm/blob/main/LICENSE
# All rights are reserved

from pathlib import Path
import sys
from lollms.app import LollmsApplication
from lollms.paths import LollmsPaths
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
import time
import io
import sys
import requests
import os
import base64
import subprocess
import time
import json
import platform
from dataclasses import dataclass
from PIL import Image, PngImagePlugin
from enum import Enum
from typing import List, Dict, Any

from ascii_colors import ASCIIColors, trace_exception
from lollms.paths import LollmsPaths
from lollms.utilities import git_pull
import subprocess
import platform


def verify_vllm(lollms_paths:LollmsPaths):
    # Clone repository

    root_dir = lollms_paths.personal_path
    shared_folder = root_dir/"shared"
    sd_folder = shared_folder / "auto_sd"
    return sd_folder.exists()
    

def install_vllm(lollms_app:LollmsApplication):
    if platform.system() == 'Windows':
        root_path = "/mnt/"+"".join(str(Path(__file__).parent).replace("\\","/").split(":"))
        if not os.path.exists('C:\\Windows\\System32\\wsl.exe'):
            if not lollms_app.YesNoMessage("No WSL is detected on your system. Do you want me to install it for you? vLLM won't be abble to work without wsl."):
                return False
            subprocess.run(['wsl', '--install', 'Ubuntu'])
        subprocess.run(['wsl', 'bash', '-c', 'cp {} ~'.format( root_path + '/install_vllm.sh')])
        subprocess.run(['wsl', 'bash', '-c', 'cp {} ~'.format( root_path + '/run_vllm.sh')])
        subprocess.run(['wsl', 'bash', '~/install_vllm.sh'])
    else:
        root_path = str(Path(__file__).parent)
        home = Path.home()
        subprocess.run(['cp {} {}'.format( root_path + '/install_vllm.sh', home)])
        subprocess.run(['cp {} {}'.format( root_path + '/run_vllm.sh', home)])
        subprocess.run(['bash', f'{home}/install_vllm.sh'])
    return True
class Service:
    def __init__(
                    self, 
                    app:LollmsApplication,
                    base_url="http://127.0.0.1:11434",
                    wait_max_retries = 5
                ):
        self.base_url = base_url
        # Get the current directory
        lollms_paths = app.lollms_paths
        self.app = app
        root_dir = lollms_paths.personal_path
        
        ASCIIColors.red("   __         __    __          __          __    __         ")
        ASCIIColors.red("  / /  ___   / /   / /   /\/\  / _\ __   __/ /   / /   /\/\  ")
        ASCIIColors.red(" / /  / _ \ / /   / /   /    \ \ \  \ \ / / /   / /   /    \ ")
        ASCIIColors.red("/ /__| (_) / /___/ /___/ /\/\ \_\ \  \ V / /___/ /___/ /\/\ \ ")
        ASCIIColors.red("\____/\___/\____/\____/\/    \/\__/___\_/\____/\____/\/    \/")
        ASCIIColors.red("                                 |_____|                     ")

        ASCIIColors.red(" Launching vllm service by vllm team")
        ASCIIColors.red(" Integration in lollms by ParisNeo")

        if not self.wait_for_service(1,False) and base_url is None:
            ASCIIColors.info("Loading vllm service")

        # run vllm
        if platform.system() == 'Windows':
            subprocess.Popen(['wsl', 'bash', '~/run_vllm.sh '])
        else:
            subprocess.Popen(['bash', f'{Path.home()}/run_vllm.sh'])

        # Wait until the service is available at http://127.0.0.1:7860/
        self.wait_for_service(max_retries=wait_max_retries)

    def wait_for_service(self, max_retries = 150, show_warning=True):
        url = f"{self.base_url}"
        # Adjust this value as needed
        retries = 0

        while retries < max_retries or max_retries<0:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print("Service is available.")
                    if self.app is not None:
                        self.app.success("vLLM Service is now available.")
                    return True
            except requests.exceptions.RequestException:
                pass

            retries += 1
            time.sleep(1)
        if show_warning:
            print("Service did not become available within the given time.\nThis may be a normal behavior as it depends on your system performance. Maybe you should wait a little more before using the vllm client as it is not ready yet\n")
            if self.app is not None:
                self.app.error("vLLM Service did not become available within the given time.")
        return False
