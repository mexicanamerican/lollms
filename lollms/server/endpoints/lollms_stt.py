"""
project: lollms_webui
file: lollms_xstt.py 
author: ParisNeo
description: 
    This module contains a set of FastAPI routes that provide information about the Lord of Large Language and Multimodal Systems (LoLLMs) Web UI
    application. These routes allow users to 

"""
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from lollms_webui import LOLLMSWebUI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from lollms.types import MSG_OPERATION_TYPE
from lollms.main_config import BaseConfig
from lollms.utilities import find_next_available_filename, output_file_path_to_url, detect_antiprompt, remove_text_from_string, trace_exception, find_first_available_file_index, add_period, PackageManager
from lollms.security import sanitize_path, validate_path, check_access
from pathlib import Path
from ascii_colors import ASCIIColors
import os
import platform

# ----------------------- Defining router and main class ------------------------------

router = APIRouter()
lollmsElfServer:LOLLMSWebUI = LOLLMSWebUI.get_instance()

class Identification(BaseModel):
    client_id: str

class ServiceListingRequest(BaseModel):
    client_id: str

@router.get("/list_stt_models")
def list_stt_models():
    if lollmsElfServer.config.headless_server_mode:
        return {"status":False,"error":"Code execution is blocked when in headless mode for obvious security reasons!"}

    if lollmsElfServer.config.host!="localhost" and lollmsElfServer.config.host!="127.0.0.1":
        return {"status":False,"error":"Code execution is blocked when the server is exposed outside for very obvious reasons!"}

    ASCIIColors.yellow("Listing voices")
    return {"voices":lollmsElfServer.stt.get_models()}


class LollmsAudio2TextRequest(BaseModel):
    wave_file_path: str
    model: str = None
    fn:str = None

@router.post("/audio2text")
async def audio2text(request: LollmsAudio2TextRequest):
    if lollmsElfServer.config.headless_server_mode:
        return {"status":False,"error":"Code execution is blocked when in headless mode for obvious security reasons!"}

    if lollmsElfServer.config.host!="localhost" and lollmsElfServer.config.host!="127.0.0.1":
        return {"status":False,"error":"Code execution is blocked when the server is exposed outside for very obvious reasons!"}

    result = lollmsElfServer.whisper.transcribe(str(request.wave_file_path))
    return PlainTextResponse(result)



@router.post("/list_stt_services")
async def list_stt_services(request: ServiceListingRequest):
    """
    Dumb endpoint that returns a static list of STT services.
    
    Args:
        request (ServiceListingRequest): The request body containing the client_id.
    
    Returns:
        List[str]: A list of STT service names.
    """
    # Validate the client_id (dumb validation for demonstration)
    check_access(lollmsElfServer, request.client_id)
    
    
    # Static list of STT services
    stt_services = [
                    {"name": "whisper", "caption":"Whisper",  "help":"Whisper local speech to text service"},
                    {"name": "openai_whisper", "caption":"Open AI Whisper STT", "help":"Open ai remote speech to text service"},
                ]
    
    return stt_services

@router.post("/get_active_stt_settings")
async def get_active_stt_settings(request: Request):
    data = await request.json()
    check_access(lollmsElfServer,data["client_id"])
    print("- Retreiving stt settings")
    if lollmsElfServer.stt is not None:
        if hasattr(lollmsElfServer.stt,"service_config"):
            return lollmsElfServer.stt.service_config.config_template.template
        else:
            return {}
    else:
        return {}

@router.post("/set_active_stt_settings")
async def set_active_stt_settings(request: Request):
    data = await request.json()
    check_access(lollmsElfServer,data["client_id"])
    settings = data["settings"]
    """
    Sets the active stt settings.

    :param request: The sttSettingsRequest object.
    :return: A JSON response with the status of the operation.
    """

    try:
        print("- Setting stt settings")
        
        if lollmsElfServer.stt is not None:
            if hasattr(lollmsElfServer.stt,"service_config"):
                lollmsElfServer.stt.service_config.update_template(settings)
                lollmsElfServer.stt.service_config.config.save_config()
                lollmsElfServer.stt.settings_updated()
                return {'status':True}
            else:
                return {'status':False}
        else:
            return {'status':False}
    except Exception as ex:
        trace_exception(ex)
        lollmsElfServer.error(ex)
        return {"status":False,"error":str(ex)}