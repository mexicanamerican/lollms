from fastapi import HTTPException
from ascii_colors import ASCIIColors
from pathlib import Path

def sanitize_path(path:str):
    if(".." in path or Path(path).is_absolute()):
        ASCIIColors.warning("Absolute database path detected")
        raise "Detected an attempt of path traversal. Are you kidding me?"
    
def sanitize_path_from_endpoint(path:str):
    if (".." in path or Path(path).is_absolute()):
        ASCIIColors.error("A suspected LFI attack detected. The path sent to the server has .. in it!")
        raise HTTPException(status_code=400, detail="Invalid path!")

