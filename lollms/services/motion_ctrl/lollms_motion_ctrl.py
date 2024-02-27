# Title LollmsMotionCtrl
# Licence: MIT
# Author : Paris Neo
# Adapted from the work of mix1009's sdwebuiapi
# check it out : https://github.com/mix1009/sdwebuiapi/tree/main
# Here is a copy of the LICENCE https://github.com/mix1009/sdwebuiapi/blob/main/LICENSE
# All rights are reserved

from pathlib import Path
import sys
from lollms.app import LollmsApplication
from lollms.paths import LollmsPaths
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
from lollms.utilities import get_conda_path, show_yes_no_dialog
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





def verify_motion_ctrl(lollms_paths:LollmsPaths):
    # Clone repository
    root_dir = lollms_paths.personal_path
    shared_folder = root_dir/"shared"
    motion_ctrl_folder = shared_folder / "auto_motion_ctrl"
    return motion_ctrl_folder.exists()
    
def install_motion_ctrl(lollms_app:LollmsApplication):
    root_dir = lollms_app.lollms_paths.personal_path
    shared_folder = root_dir/"shared"
    motion_ctrl_folder = shared_folder / "auto_motion_ctrl"
    if motion_ctrl_folder.exists():
        if not show_yes_no_dialog("warning!","I have detected that there is a previous installation of motion ctrl.\nShould I remove it and continue installing?"):
            return
        else:
            motion_ctrl_folder.unlink(True)
    subprocess.run(["git", "clone", "https://github.com/ParisNeo/MotionCtrl.git", str(motion_ctrl_folder)])
    import conda.cli
    env_name = "MotionCtrl"

    conda.cli.main('conda', 'create', '--name', env_name, 'python=3.10', '--yes')
    # Replace 'your_env_name' with the name of the environment you created
    activate_env_command = f"conda activate {env_name} && "
    pip_install_command = "pip install -r " + str(motion_ctrl_folder) + "/requirements.txt"

    # Run the combined command
    subprocess.run(activate_env_command + pip_install_command, shell=True, executable='/bin/bash')    
    #pip install -r requirements.txt    
    ASCIIColors.green("Motion ctrl installed successfully")


def get_motion_ctrl(lollms_paths:LollmsPaths):
    root_dir = lollms_paths.personal_path
    shared_folder = root_dir/"shared"
    motion_ctrl_folder = shared_folder / "auto_motion_ctrl"
    motion_ctrl_script_path = motion_ctrl_folder / "lollms_motion_ctrl.py"
    git_pull(motion_ctrl_folder)
    
    if motion_ctrl_script_path.exists():
        ASCIIColors.success("lollms_motion_ctrl found.")
        ASCIIColors.success("Loading source file...",end="")
        # use importlib to load the module from the file path
        from lollms.services.motion_ctrl.lollms_motion_ctrl import LollmsMotionCtrl
        ASCIIColors.success("ok")
        return LollmsMotionCtrl


class LollmsMotionCtrl:
    has_controlnet = False
    def __init__(
                    self, 
                    app:LollmsApplication, 
                    wm = "Artbot", 
                    max_retries=50,
                    sampler="Euler a",
                    steps=20,               
                    use_https=False,
                    username=None,
                    password=None,
                    auto_motion_ctrl_base_url=None,
                    share=False,
                    wait_for_service=True
                    ):
        if auto_motion_ctrl_base_url=="" or auto_motion_ctrl_base_url=="http://127.0.0.1:7860":
            auto_motion_ctrl_base_url = None
        # Get the current directory
        lollms_paths = app.lollms_paths
        self.app = app
        root_dir = lollms_paths.personal_path
        
        self.wm = wm
        # Store the path to the script
        if auto_motion_ctrl_base_url is None:
            self.auto_motion_ctrl_base_url = "http://127.0.0.1:7860"
            if not verify_motion_ctrl(lollms_paths):
                install_motion_ctrl(app.lollms_paths)
        else:
            self.auto_motion_ctrl_base_url = auto_motion_ctrl_base_url

        self.auto_motion_ctrl_url = self.auto_motion_ctrl_base_url+"/sdapi/v1"
        shared_folder = root_dir/"shared"
        self.motion_ctrl_folder = shared_folder / "motion_ctrl"
        self.output_dir = root_dir / "outputs/motion_ctrl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

       
        ASCIIColors.red("                                                       ")
        ASCIIColors.red(" __    _____ __    __    _____ _____       _____ ____  ")
        ASCIIColors.red("|  |  |     |  |  |  |  |     |   __|     |   __|    \ ")
        ASCIIColors.red("|  |__|  |  |  |__|  |__| | | |__   |     |__   |  |  |")
        ASCIIColors.red("|_____|_____|_____|_____|_|_|_|_____|_____|_____|____/ ")
        ASCIIColors.red("                                    |_____|            ")

        ASCIIColors.red(" Forked from TencentARC's MotionCtrl api")
        ASCIIColors.red(" Integration in lollms by ParisNeo")
        motion_ctrl_folder = shared_folder / "auto_motion_ctrl"
        env_name = "MotionCtrl"

        if not self.wait_for_service(1,False) and auto_motion_ctrl_base_url is None:
            ASCIIColors.info("Loading lollms_motion_ctrl")
            os.environ['motion_ctrl_WEBUI_RESTARTING'] = '1' # To forbid sd webui from showing on the browser automatically

            # Get the current operating system
            os_name = platform.system()
            conda_path = get_conda_path()

            # Construct the command to activate the environment and start the server
            if os_name == "Windows":
                activate_env_command = f"call {conda_path}\\activate {env_name} && "
            else:  # Linux or macOS
                activate_env_command = f"source {conda_path}/activate {env_name} && "
            
            start_server_command = f"python -m app --share"

            # Run the combined command
            if os_name == "Windows":
                subprocess.run(activate_env_command + start_server_command, shell=True, cwd=motion_ctrl_folder)
            else:  # Linux or macOS
                subprocess.run(activate_env_command + start_server_command, shell=True, executable='/bin/bash', cwd=motion_ctrl_folder)


        # Wait until the service is available at http://127.0.0.1:7860/
        if wait_for_service:
            self.wait_for_service(max_retries=max_retries)
        else:
            ASCIIColors.warning("We are not waiting for the MotionCtrl service to be up.\nThis means that you may need to wait a bit before you can use it.")

        self.default_sampler = sampler
        self.default_steps = steps

        self.session = requests.Session()

        if username and password:
            self.set_auth(username, password)
        else:
            self.check_controlnet()
