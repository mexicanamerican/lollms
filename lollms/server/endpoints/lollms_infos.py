"""
project: lollms
file: lollms_infos.py 
author: ParisNeo
description: 
    This module contains a set of FastAPI routes that provide information about the Lord of Large Language and Multimodal Systems (LoLLMs) Web UI
    application. These routes are specific to handling lollms information like version and configuration

"""
from fastapi import APIRouter, Request
import pkg_resources
from lollms.server.elf_server import LOLLMSElfServer
from ascii_colors import ASCIIColors
from lollms.utilities import load_config
from pathlib import Path
from typing import List
import psutil
import os
# ----------------------- Defining router and main class ------------------------------
router = APIRouter()
lollmsElfServer = LOLLMSElfServer.get_instance()

@router.get("/get_lollms_version")
@router.get("/version")
def get_version():
    """
    Get the version of the lollms package.

    Returns:
        dict: A dictionary containing the version of the lollms package.
    """
    version = pkg_resources.get_distribution('lollms').version
    ASCIIColors.yellow("Lollms version: " + version)
    return {"version": version}


@router.get("/get_server_address")
def get_server_address(request:Request):
    """
    Get the server address for the current request.

    Returns:
        str: The server address without the endpoint path.
    """    
    server_address = request.url_for("get_server_address")
    srv_addr = "/".join(str(server_address).split("/")[:-1])
    ASCIIColors.yellow(server_address)
    return srv_addr



def open_folder(path: str):
    path = Path(path).absolute()
    if not path.is_dir():
        raise FileNotFoundError(f"The provided path '{path}' is not a directory.")
    os.startfile(path)
    
@router.get("/open_folder")
async def open_folder(request: Request, path: str):
    if Path(path).exists() and Path(path).is_dir():
        os.startfile(path)
        return {"message": "Folder opened successfully."}
    else:
        return {"message": "Invalid folder path or the folder does not exist."}
