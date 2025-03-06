from fastapi import APIRouter, HTTPException, Request
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

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 512
    height: int = 512

class ImageResponse(BaseModel):
    image: str

class ServiceListingRequest(BaseModel):
    client_id: str

@router.post("/list_tti_services")
async def list_tti_services(request: ServiceListingRequest):
    """
    Dumb endpoint that returns a static list of TTI services.
    
    Args:
        request (ServiceListingRequest): The request body containing the client_id.
    
    Returns:
        List[str]: A list of TTV service names.
    """
    # Validate the client_id (dumb validation for demonstration)
    check_access(lollmsElfServer, request.client_id)
    
    
    # Static list of TTV services
    ttv_services = [
                    {"name": "diffusers", "caption":"Diffusers",  "help":"Diffusers text to image service"},
                    {"name": "diffusers_client", "caption":"Diffusers Client",  "help":"Diffusers Client text to image service"},
                    {"name": "autosd", "caption":"AUTO1111's SD",  "help":"AUTO1111's SD text to image service"},
                    {"name": "dall-e", "caption":"Open AI's DALL-E",  "help":"Open AI's DALL-E text to image service"},
                    {"name": "midjourney", "caption":"Midjourney",  "help":"Midjourney text to image service"},
                    {"name": "comfyui", "caption":"Comfyui",  "help":"Comfyui text to image service"},
                    {"name": "fooocus", "caption":"Fooocus",  "help":"Fooocus text to image service"},
                ]
    return ttv_services

@router.post("/generate_image", response_model=ImageResponse)
async def generate_image(request: ImageRequest):
    try:
        import uuid

        filename = f"remote_gen_{uuid.uuid4().hex[:8]}.png"
        output_path = lollmsElfServer.lollms_paths.personal_outputs_path/filename        
        # Call the build_image function
        result = build_image(
            request.prompt,
            request.negative_prompt,
            request.width,
            request.height,
            return_format="path",
            output_path=output_path
        )

        # Check if image generation was successful
        if result is None:
            raise HTTPException(status_code=500, detail="Image generation failed")

        # Open the image file
        with Image.open(result) as img:
            # Convert the image to RGB mode (in case it's RGBA)
            img = img.convert("RGB")
            
            # Save the image to a bytes buffer
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            
            # Encode the image as base64
            img_base64 = b64encode(buf.getvalue()).decode()

        # Return the base64 encoded image
        return ImageResponse(image=img_base64)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def build_image(prompt, negative_prompt, width, height, return_format="markdown", output_path =None):
    try:
        import uuid
        if output_path is None:
            filename = f"remote_gen_{uuid.uuid4().hex[:8]}.png"
            output_path = lollmsElfServer.lollms_paths.personal_outputs_path/filename

        if lollmsElfServer.tti is not None:
            file, infos = lollmsElfServer.tti.paint(
                prompt,
                negative_prompt,
                width=width,
                height=height,
                output_path=output_path
            )

        file = str(file)

        if return_format == "path":
            return file
        else:
            return None  # Handle other return formats if needed
    except Exception as ex:
        # Log the exception
        trace_exception(ex)
        print(f"Error in build_image: {str(ex)}")
        return None

@router.post("/get_active_tti_settings")
async def get_active_tti_settings(request: ServiceListingRequest):
    check_access(lollmsElfServer,request.client_id)
    ASCIIColors.info("Retreiving tti settings")
    if lollmsElfServer.tti is not None:
        if hasattr(lollmsElfServer.tti,"service_config"):
            return lollmsElfServer.tti.service_config.config_template.template
        else:
            return {}
    else:
        return {}


@router.post("/set_active_tti_settings")
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
        print("- Setting tti settings")
        
        if lollmsElfServer.tti is not None:
            if hasattr(lollmsElfServer.tti,"service_config"):
                lollmsElfServer.tti.service_config.update_template(settings)
                lollmsElfServer.tti.service_config.config.save_config()
                lollmsElfServer.tti.settings_updated()
                return {'status':True}
            else:
                return {'status':False}
        else:
            return {'status':False}
    except Exception as ex:
        trace_exception(ex)
        lollmsElfServer.error(ex)
        return {"status":False,"error":str(ex)}
