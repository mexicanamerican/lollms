from fastapi import HTTPException
from ascii_colors import ASCIIColors
from pathlib import Path
from typing import List
import os

def sanitize_path(path:str, allow_absolute_path:bool=False, error_text="Absolute database path detected", exception_text="Detected an attempt of path traversal. Are you kidding me?"):
    if(".." in path):
        ASCIIColors.warning(error_text)
        raise exception_text
    if (not allow_absolute_path) and Path(path).is_absolute():
        ASCIIColors.warning(error_text)
        raise exception_text
    
def sanitize_path_from_endpoint(path:str, error_text="A suspected LFI attack detected. The path sent to the server has .. in it!", exception_text="Invalid path!"):
    if (".." in path or Path(path).is_absolute()):
        ASCIIColors.error(error_text)
        raise HTTPException(status_code=400, detail=exception_text)

def forbid_remote_access(lollmsElfServer):
    if lollmsElfServer.config.host!="localhost" and lollmsElfServer.config.host!="127.0.0.1":
        return {"status":False,"error":"Code execution is blocked when the server is exposed outside for very obvious reasons!"}


def validate_path(path, allowed_paths:List[str|Path]):
    # Convert the path to an absolute path
    abs_path = os.path.realpath(str(path))

    # Iterate over the allowed paths
    for allowed_path in allowed_paths:
        # Convert the allowed path to an absolute path
        abs_allowed_path = os.path.realpath(allowed_path)

        # Check if the absolute path starts with the absolute allowed path
        if abs_path.startswith(abs_allowed_path):
            return True

    # If the path is not within any of the allowed paths, return False
    return False
