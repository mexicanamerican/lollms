from fastapi import APIRouter, HTTPException
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


# Define a Pydantic model for the request body
class TTVServiceRequest(BaseModel):
    client_id: str

@router.post("/list_ttv_services", response_model=List[str])
async def list_ttv_services(request: TTVServiceRequest):
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
    ttv_services = ["novita_ai", "cog_video_x", "diffusers", "lumalab"]
    
    return ttv_services