from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
from base64 import b64encode
import io
from PIL import Image
from fastapi import APIRouter
from lollms_webui import LOLLMSWebUI
from pydantic import BaseModel
from typing import List
from ascii_colors import trace_exception
from lollms.security import check_access

router = APIRouter()
lollmsElfServer = LOLLMSWebUI.get_instance()

class ServiceListingRequest(BaseModel):
    client_id: str


# Define a Pydantic model for the request body

@router.post("/list_ttm_services")
async def list_ttm_services(request: ServiceListingRequest):
    """
    Dumb endpoint that returns a static list of TTM services.
    
    Args:
        request (ServiceListingRequest): The request body containing the client_id.
    
    Returns:
        List[str]: A list of TTM service names.
    """
    # Validate the client_id (dumb validation for demonstration)
    check_access(lollmsElfServer, request.client_id)
    
    
    # Static list of TTM services
    ttm_services = [
                    {"name": "suno", "caption":"Suno AI",  "help":"Suno ai"},
                    {"name": "music_gen", "caption":"Music Gen",  "help":"Music Gen"},
                ]
    return ttm_services


@router.post("/get_active_ttm_settings")
async def get_active_ttm_settings(request: Request):
    data = await request.json()
    check_access(lollmsElfServer,data["client_id"])
    print("- Retreiving ttm settings")
    if lollmsElfServer.ttm is not None:
        if hasattr(lollmsElfServer.ttm,"service_config"):
            return lollmsElfServer.ttm.service_config.config_template.template
        else:
            return {}
    else:
        return {}

@router.post("/set_active_ttm_settings")
async def set_active_ttm_settings(request: Request):
    data = await request.json()
    check_access(lollmsElfServer,data["client_id"])
    settings = data["settings"]
    """
    Sets the active ttm settings.

    :param request: The Request object.
    :return: A JSON response with the status of the operation.
    """

    try:
        print("- Setting ttm settings")
        
        if lollmsElfServer.ttm is not None:
            if hasattr(lollmsElfServer.ttm,"service_config"):
                lollmsElfServer.ttm.service_config.update_template(settings)
                lollmsElfServer.ttm.service_config.config.save_config()
                lollmsElfServer.ttm.settings_updated()
                return {'status':True}
            else:
                return {'status':False}
        else:
            return {'status':False}
    except Exception as ex:
        trace_exception(ex)
        lollmsElfServer.error(ex)
        return {"status":False,"error":str(ex)}