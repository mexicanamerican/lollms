"""
project: lollms
file: lollms_files_events.py 
author: ParisNeo
description: 
    Events related to socket io personality events

"""
from fastapi import APIRouter, Request
from fastapi import HTTPException
from pydantic import BaseModel
import pkg_resources
from lollms.server.elf_server import LOLLMSElfServer
from fastapi.responses import FileResponse
from lollms.binding import BindingBuilder, InstallOption
from ascii_colors import ASCIIColors
from lollms.personality import MSG_TYPE, AIPersonality
from lollms.utilities import load_config, trace_exception, gc, terminate_thread, run_async
from pathlib import Path
from typing import List
import socketio
from functools import partial
from datetime import datetime
import os

router = APIRouter()
lollmsElfServer = LOLLMSElfServer.get_instance()


# ----------------------------------- events -----------------------------------------
def add_events(sio:socketio):
            
    @sio.on('get_personality_files')
    def get_personality_files(sid, data):
        client_id = sid
        lollmsElfServer.connections[client_id]["generated_text"]       = ""
        lollmsElfServer.connections[client_id]["cancel_generation"]    = False
        
        try:
            lollmsElfServer.personality.setCallback(partial(lollmsElfServer.process_chunk,client_id = client_id))
        except Exception as ex:
            trace_exception(ex)        

    @sio.on('send_file_chunk')
    def send_file_chunk(sid, data):
        client_id = sid
        filename = data['filename']
        chunk = data['chunk']
        offset = data['offset']
        is_last_chunk = data['isLastChunk']
        chunk_index = data['chunkIndex']
        path:Path = lollmsElfServer.lollms_paths.personal_uploads_path / lollmsElfServer.personality.personality_folder_name
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / data["filename"]
        # Save the chunk to the server or process it as needed
        # For example:
        if chunk_index==0:
            with open(file_path, 'wb') as file:
                file.write(chunk)
        else:
            with open(file_path, 'ab') as file:
                file.write(chunk)

        if is_last_chunk:
            ASCIIColors.success('File received and saved successfully')
            if lollmsElfServer.personality.processor:
                result = lollmsElfServer.personality.processor.add_file(file_path, partial(lollmsElfServer.process_chunk, client_id=client_id))
            else:
                result = lollmsElfServer.personality.add_file(file_path, partial(lollmsElfServer.process_chunk, client_id=client_id))

            ASCIIColors.success('File processed successfully')
            run_async(partial(sio.emit,'file_received', {'status': True, 'filename': filename}))
        else:
            # Request the next chunk from the client
            run_async(partial(sio.emit,'request_next_chunk', {'offset': offset + len(chunk)}))

    @sio.on('execute_command')
    def execute_command(sid, data):
        client_id = sid
        command = data["command"]
        parameters = data["parameters"]
        if lollmsElfServer.personality.processor is not None:
            lollmsElfServer.start_time = datetime.now()
            lollmsElfServer.personality.processor.callback = partial(lollmsElfServer.process_chunk, client_id=client_id)
            lollmsElfServer.personality.processor.execute_command(command, parameters)
        else:
            lollmsElfServer.warning("Non scripted personalities do not support commands",client_id=client_id)
        lollmsElfServer.close_message(client_id)
    