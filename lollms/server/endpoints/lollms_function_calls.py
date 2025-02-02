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
import os
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


