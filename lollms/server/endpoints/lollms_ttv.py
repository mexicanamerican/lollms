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
from ascii_colors import ASCIIColors

router = APIRouter()
lollmsElfServer = LOLLMSWebUI.get_instance()


# Define a Pydantic model for the request body
class ServiceListingRequest(BaseModel):
    client_id: str

class TTVServiceGetConfigRequest(BaseModel):
    client_id: str
    service_name: str

class TTVServiceSetConfigRequest(BaseModel):
    client_id: str
    service_name: str


@router.post("/list_ttv_services")
async def list_ttv_services(request: ServiceListingRequest):
    """
    Dumb endpoint that returns a static list of TTV services.
    
    Args:
        request (TTVServiceRequest): The request body containing the client_id.
    
    Returns:
        List[str]: A list of TTV service names.
    """
    # Validate the client_id (dumb validation for demonstration)
    check_access(lollmsElfServer, request.client_id)
    
    
    # Static list of TTV services
    ttv_services = [
                    {"name": "novita_ai", "caption":"Novita AI",  "help":"Novita ai text to video services"},
                    {"name": "diffusers", "caption":"Diffusers", "help":"Diffusers based Local text to video services"},
                    {"name": "lumalabs", "caption":"Luma labs", "help":"Luma labs text to video services"},
                    {"name": "cog_video_x", "caption":"Cog Video", "help":"Cog video"},
                ]
    
    return ttv_services

@router.post("/get_active_ttv_settings")
async def get_active_ttv_settings(request: Request):
    data = await request.json()
    check_access(lollmsElfServer,data["client_id"])
    print("- Retreiving ttv settings")
    if lollmsElfServer.ttv is not None:
        if hasattr(lollmsElfServer.ttv,"service_config"):
            return lollmsElfServer.ttv.service_config.config_template.template
        else:
            return {}
    else:
        return {}


@router.post("/set_active_ttv_settings")
async def set_active_ttv_settings(request: Request):
    data = await request.json()
    check_access(lollmsElfServer,data["client_id"])
    settings = data["settings"]
    """
    Sets the active ttv settings.

    :param request: The ttvSettingsRequest object.
    :return: A JSON response with the status of the operation.
    """

    try:
        print("- Setting ttv settings")
        
        if lollmsElfServer.ttv is not None:
            if hasattr(lollmsElfServer.ttv,"service_config"):
                lollmsElfServer.ttv.service_config.update_template(settings)
                lollmsElfServer.ttv.service_config.config.save_config()
                lollmsElfServer.ttv.settings_updated()
                return {'status':True}
            else:
                return {'status':False}
        else:
            return {'status':False}
    except Exception as ex:
        trace_exception(ex)
        lollmsElfServer.error(ex)
        return {"status":False,"error":str(ex)}