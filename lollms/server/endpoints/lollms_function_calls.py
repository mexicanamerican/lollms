"""
project: lollms
file: lollms_binding_infos.py 
author: ParisNeo
description: 
    This module contains a set of FastAPI routes that provide information about the Lord of Large Language and Multimodal Systems (LoLLMs) Web UI
    application. These routes are specific to bindings

"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
import pkg_resources
from lollms.server.elf_server import LOLLMSElfServer
from lollms.binding import BindingBuilder, InstallOption
from ascii_colors import ASCIIColors
from lollms.utilities import load_config, trace_exception, gc
from lollms.security import sanitize_path_from_endpoint, sanitize_path, check_access
from lollms.security import check_access
from pathlib import Path
from typing import List, Any
import json
import yaml
# ----------------------------------- Personal files -----------------------------------------

class ClientAuthentication(BaseModel):
    client_id: str  = Field(...)

class ReloadBindingParams(BaseModel):
    binding_name: str = Field(..., min_length=1, max_length=50)

class BindingInstallParams(BaseModel):
    client_id: str
    name: str = Field(..., min_length=1, max_length=50)


# ----------------------- Defining router and main class ------------------------------
router = APIRouter()
lollmsElfServer = LOLLMSElfServer.get_instance()


# ----------------------------------- Endpoints -----------------------------------------


@router.get("/list_function_calls")
async def list_function_calls():
    """List all available function calls in the functions zoo"""
    functions_zoo_path = lollmsElfServer.paths.functions_zoo_path
    function_calls = []
    
    for fn_dir in functions_zoo_path.iterdir():
        if fn_dir.is_dir():
            yaml_file = fn_dir / "config.yaml"
            py_file = fn_dir / "function.py"
            
            if yaml_file.exists() and py_file.exists():
                try:
                    with open(yaml_file, "r") as f:
                        config = yaml.safe_load(f)
                        function_info = {
                            "name": config.get("name", fn_dir.name),
                            "description": config.get("description", ""),
                            "parameters": config.get("parameters", {}),
                            "returns": config.get("returns", {}),
                            "examples": config.get("examples", []),
                            "author": config.get("author", "Unknown"),
                            "version": config.get("version", "1.0.0")
                        }
                        function_calls.append(function_info)
                except Exception as e:
                    ASCIIColors.error(f"Error loading function {fn_dir.name}: {e}")
    
    return {"function_calls": function_calls}

@router.get("/list_mounted_function_calls")
async def list_mounted_function_calls():
    """List currently mounted function calls"""
    mounted = [fc["name"] for fc in lollmsElfServer.config.mounted_function_calls if fc["mounted"]]
    return {"mounted_function_calls": mounted}

@router.post("/mount_function_call")
async def mount_function_call(request: Request):
    """Mount a function call to make it available to the LLM"""
    data = await request.json()
    client_id = data.get("client_id")
    function_name = data.get("function_name")

    if not check_access(client_id, lollmsElfServer):
        raise HTTPException(status_code=403, detail="Access denied")

    # Validate function exists
    fn_dir = lollmsElfServer.paths.functions_zoo_path / function_name
    if not fn_dir.exists() or not (fn_dir / "config.yaml").exists() or not (fn_dir / "function.py").exists():
        raise HTTPException(status_code=404, detail="Function not found")

    # Check if already mounted
    for fc in lollmsElfServer.config.mounted_function_calls:
        if fc["name"] == function_name:
            if fc["mounted"]:
                return {"status": False, "message": "Function already mounted"}
            fc["mounted"] = True
            lollmsElfServer.config.save_config()
            return {"status": True, "message": "Function mounted"}

    # Add new entry
    lollmsElfServer.config.mounted_function_calls.append({
        "name": function_name,
        "mounted": True
    })
    lollmsElfServer.config.save_config()
    
    return {"status": True, "message": "Function mounted successfully"}

@router.post("/unmount_function_call")
async def unmount_function_call(request: Request):
    """Unmount a function call to remove it from LLM's availability"""
    data = await request.json()
    client_id = data.get("client_id")
    function_name = data.get("function_name")

    if not check_access(client_id, lollmsElfServer):
        raise HTTPException(status_code=403, detail="Access denied")

    # Find and update the function call
    found = False
    for fc in lollmsElfServer.config.mounted_function_calls:
        if fc["name"] == function_name and fc["mounted"]:
            fc["mounted"] = False
            found = True
            break
    
    if not found:
        raise HTTPException(status_code=404, detail="Function not mounted")

    lollmsElfServer.config.save_config()
    return {"status": True, "message": "Function unmounted successfully"}