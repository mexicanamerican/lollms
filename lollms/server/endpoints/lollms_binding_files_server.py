"""
project: lollms
file: lollms_binding_files_server.py 
author: ParisNeo
description: 
    This module contains a set of FastAPI routes that provide information about the Lord of Large Language and Multimodal Systems (LoLLMs) Web UI
    application. These routes are specific to serving files

"""
from fastapi import APIRouter, Request
from fastapi import HTTPException
from pydantic import BaseModel
import pkg_resources
from lollms.server.elf_server import LOLLMSElfServer
from fastapi.responses import FileResponse
from lollms.binding import BindingBuilder, InstallOption
from ascii_colors import ASCIIColors
from lollms.utilities import load_config, trace_exception, gc
from pathlib import Path
from typing import List
import os

# ----------------------- Defining router and main class ------------------------------
router = APIRouter()
lollmsElfServer = LOLLMSElfServer.get_instance()


# ----------------------------------- Personal files -----------------------------------------
@router.get("/user_infos/{path:path}")
async def serve_user_infos(path: str):
    """
    Serve user information file.

    Args:
        filename (str): The name of the file to serve.

    Returns:
        FileResponse: The file response containing the requested file.
    """
    file_path = lollmsElfServer.lollms_paths.personal_user_infos_path / path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))

# ----------------------------------- Lollms zoos -----------------------------------------


@router.get("/bindings/{path:path}")
async def serve_bindings(path: str):
    """
    Serve personalities file.

    Args:
        path (str): The path of the bindings file to serve.

    Returns:
        FileResponse: The file response containing the requested bindings file.
    """
    file_path = lollmsElfServer.lollms_paths.bindings_zoo_path / path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))

@router.get("/personalities/{path:path}")
async def serve_personalities(path: str):
    """
    Serve personalities file.

    Args:
        path (str): The path of the personalities file to serve.

    Returns:
        FileResponse: The file response containing the requested personalities file.
    """
    if "custom_personalities" in path:
        file_path = lollmsElfServer.lollms_paths.custom_personalities_path / "/".join(str(path).split("/")[1:])
    else:
        file_path = lollmsElfServer.lollms_paths.personalities_zoo_path / path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))

@router.get("/extensions/{path:path}")
async def serve_extensions(path: str):
    """
    Serve personalities file.

    Args:
        path (str): The path of the extensions file to serve.

    Returns:
        FileResponse: The file response containing the requested extensions file.
    """
    file_path = lollmsElfServer.lollms_paths.extensions_zoo_path / path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))

# ----------------------------------- Services -----------------------------------------

@router.get("/audio/{path:path}")
async def serve_audio(path: str):
    """
    Serve audio file.

    Args:
        filename (str): The name of the audio file to serve.

    Returns:
        FileResponse: The file response containing the requested audio file.
    """
    root_dir = lollmsElfServer.lollms_paths.personal_outputs_path
    file_path = Path(root_dir)/ 'audio_out' / path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))


@router.get("/images/{path:path}")
async def serve_images(path: str):
    """
    Serve image file.

    Args:
        filename (str): The name of the image file to serve.

    Returns:
        FileResponse: The file response containing the requested image file.
    """
    root_dir = Path(os.getcwd())/ "images/"
    file_path = root_dir / path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))


# ----------------------------------- User content -----------------------------------------

@router.get("/outputs/{path:path}")
async def serve_outputs(path: str):
    """
    Serve image file.

    Args:
        filename (str): The name of the output file to serve.

    Returns:
        FileResponse: The file response containing the requested output file.
    """
    root_dir = lollmsElfServer.lollms_paths.personal_outputs_path
    root_dir.mkdir(exist_ok=True, parents=True)
    file_path = root_dir / path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))



@router.get("/data/{path:path}")
async def serve_data(path: str):
    """
    Serve image file.

    Args:
        filename (str): The name of the data file to serve.

    Returns:
        FileResponse: The file response containing the requested data file.
    """
    root_dir = lollmsElfServer.lollms_paths.personal_path / "data"
    root_dir.mkdir(exist_ok=True, parents=True)
    file_path = root_dir / path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))




@router.get("/help/{path:path}")
async def serve_help(path: str):
    """
    Serve image file.

    Args:
        filename (str): The name of the data file to serve.

    Returns:
        FileResponse: The file response containing the requested data file.
    """
    root_dir = Path(os.getcwd())
    file_path = root_dir/'help/' / path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))



@router.get("/uploads/{path:path}")
async def serve_uploads(path: str):
    """
    Serve image file.

    Args:
        filename (str): The name of the uploads file to serve.

    Returns:
        FileResponse: The file response containing the requested uploads file.
    """
    root_dir = lollmsElfServer.lollms_paths.personal_path / "uploads"
    root_dir.mkdir(exist_ok=True, parents=True)
    file_path = root_dir / path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))

