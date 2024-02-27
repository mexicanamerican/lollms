"""
project: lollms_webui
file: lollms_discussion.py 
author: ParisNeo
description: 
    This module contains a set of FastAPI routes that provide information about the Lord of Large Language and Multimodal Systems (LoLLMs) Web UI
    application. These routes allow users to manipulate the discussion elements.

"""
from fastapi import APIRouter, Request
from lollms_webui import LOLLMSWebUI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from lollms.types import MSG_TYPE
from lollms.utilities import detect_antiprompt, remove_text_from_string, trace_exception
from lollms.security import sanitize_path
from ascii_colors import ASCIIColors
from lollms.databases.discussions_database import DiscussionsDB, Discussion
from typing import List

from safe_store.text_vectorizer import TextVectorizer, VectorizationMethod, VisualizationMethod
import tqdm
from pathlib import Path
class GenerateRequest(BaseModel):
    text: str

class DatabaseSelectionParameters(BaseModel):
    name: str

class EditTitleParameters(BaseModel):
    client_id: str
    title: str
    id: int

class MakeTitleParameters(BaseModel):
    id: int

class DeleteDiscussionParameters(BaseModel):
    client_id: str
    id: int

# ----------------------- Defining router and main class ------------------------------

router = APIRouter()
lollmsElfServer:LOLLMSWebUI = LOLLMSWebUI.get_instance()


@router.get("/list_discussions")
def list_discussions():
    discussions = lollmsElfServer.db.get_discussions()
    return discussions


@router.get("/list_databases")
async def list_databases():
   """List all the personal databases in the LoLLMs server."""
   # Retrieve the list of database names
   databases = [f.name for f in lollmsElfServer.lollms_paths.personal_discussions_path.iterdir() if f.is_dir() and (f/"database.db").exists()]
   # Return the list of database names
   return databases


@router.post("/select_database")
def select_database(data:DatabaseSelectionParameters):
    sanitize_path(data.name)
    print(f'Selecting database {data.name}')
    # Create database object
    lollmsElfServer.db = DiscussionsDB(lollmsElfServer.lollms_paths, data.name)
    ASCIIColors.info("Checking discussions database... ",end="")
    lollmsElfServer.db.create_tables()
    lollmsElfServer.db.add_missing_columns()
    lollmsElfServer.config.discussion_db_name = data.name
    ASCIIColors.success("ok")

    if lollmsElfServer.config.auto_save:
        lollmsElfServer.config.save_config()
    
    return {"status":True}


@router.post("/export_discussion")
def export_discussion():
    return {"discussion_text":lollmsElfServer.get_discussion_to()}


class DiscussionEditTitle(BaseModel):
    client_id: str
    title: str
    id: int

@router.post("/edit_title")
async def edit_title(discussion_edit_title: DiscussionEditTitle):
    try:
        client_id = discussion_edit_title.client_id
        title = discussion_edit_title.title
        discussion_id = discussion_edit_title.id
        lollmsElfServer.session.get_client(client_id).discussion = Discussion(discussion_id, lollmsElfServer.db)
        lollmsElfServer.session.get_client(client_id).discussion.rename(title)
        return {'status':True}
    except Exception as ex:
        trace_exception(ex)
        lollmsElfServer.error(ex)
        return {"status":False,"error":str(ex)}

class DiscussionTitle(BaseModel):
    id: int
    
@router.post("/make_title")
async def make_title(discussion_title: DiscussionTitle):
    try:
        ASCIIColors.info("Making title")
        discussion_id = discussion_title.id
        discussion = Discussion(discussion_id, lollmsElfServer.db)
        title = lollmsElfServer.make_discussion_title(discussion)
        discussion.rename(title)
        return {'status':True, 'title':title}
    except Exception as ex:
        trace_exception(ex)
        lollmsElfServer.error(ex)
        return {"status":False,"error":str(ex)}
    
    
@router.get("/export")
def export():
    return lollmsElfServer.db.export_to_json()



class DiscussionDelete(BaseModel):
    client_id: str
    id: int

@router.post("/delete_discussion")
async def delete_discussion(discussion: DiscussionDelete):
    """
    Executes Python code and returns the output.

    :param request: The HTTP request object.
    :return: A JSON response with the status of the operation.
    """

    try:

        client_id           = discussion.client_id
        discussion_id       = discussion.id
        lollmsElfServer.session.get_client(client_id).discussion = Discussion(discussion_id, lollmsElfServer.db)
        lollmsElfServer.session.get_client(client_id).discussion.delete_discussion()
        lollmsElfServer.session.get_client(client_id).discussion = None
        return {'status':True}
    except Exception as ex:
        trace_exception(ex)
        lollmsElfServer.error(ex)
        return {"status":False,"error":str(ex)}
    
    
# ----------------------------- import/export --------------------
class DiscussionExport(BaseModel):
    discussion_ids: List[int]
    export_format: str

@router.post("/export_multiple_discussions")
async def export_multiple_discussions(discussion_export: DiscussionExport):
    try:
        discussion_ids = discussion_export.discussion_ids
        export_format = discussion_export.export_format

        if export_format=="json":
            discussions = lollmsElfServer.db.export_discussions_to_json(discussion_ids)
        elif export_format=="markdown":
            discussions = lollmsElfServer.db.export_discussions_to_markdown(discussion_ids)
        else:
            discussions = lollmsElfServer.db.export_discussions_to_markdown(discussion_ids)
        return discussions
    except Exception as ex:
        trace_exception(ex)
        lollmsElfServer.error(ex)
        return {"status":False,"error":str(ex)}


class DiscussionInfo(BaseModel):
    id: int
    content: str

class DiscussionImport(BaseModel):
    jArray: List[DiscussionInfo]

@router.post("/import_multiple_discussions")
async def import_multiple_discussions(discussion_import: DiscussionImport):
    try:
        discussions = discussion_import.jArray
        lollmsElfServer.db.import_from_json(discussions)
        return discussions
    except Exception as ex:
        trace_exception(ex)
        lollmsElfServer.error(ex)
        return {"status":False,"error":str(ex)}
