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
from lollms.types import SENDER_TYPES
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
        lollmsElfServer.cancel_gen = False
        lollmsElfServer.connections[client_id]["generated_text"]=""
        lollmsElfServer.connections[client_id]["cancel_generation"]=False
        lollmsElfServer.connections[client_id]["continuing"]=False
        lollmsElfServer.connections[client_id]["first_chunk"]=True
        
        if not lollmsElfServer.model:
            ASCIIColors.error("Model not selected. Please select a model")
            lollmsElfServer.error("Model not selected. Please select a model", client_id=client_id)
            return

        if not lollmsElfServer.busy:
            if lollmsElfServer.connections[client_id]["current_discussion"] is None:
                if lollmsElfServer.db.does_last_discussion_have_messages():
                    lollmsElfServer.connections[client_id]["current_discussion"] = lollmsElfServer.db.create_discussion()
                else:
                    lollmsElfServer.connections[client_id]["current_discussion"] = lollmsElfServer.db.load_last_discussion()

            ump = lollmsElfServer.config.discussion_prompt_separator +lollmsElfServer.config.user_name.strip() if lollmsElfServer.config.use_user_name_in_discussions else lollmsElfServer.personality.user_message_prefix
            message = lollmsElfServer.connections[client_id]["current_discussion"].add_message(
                message_type    = MSG_TYPE.MSG_TYPE_FULL.value,
                sender_type     = SENDER_TYPES.SENDER_TYPES_USER.value,
                sender          = ump.replace(lollmsElfServer.config.discussion_prompt_separator,"").replace(":",""),
                content="",
                metadata=None,
                parent_message_id=lollmsElfServer.message_id
            )
            lollmsElfServer.busy=True

            client_id = sid
            command = data["command"]
            parameters = data["parameters"]
            lollmsElfServer.prepare_reception(client_id)
            if lollmsElfServer.personality.processor is not None:
                lollmsElfServer.start_time = datetime.now()
                lollmsElfServer.personality.processor.callback = partial(lollmsElfServer.process_chunk, client_id=client_id)
                lollmsElfServer.personality.processor.execute_command(command, parameters)
            else:
                lollmsElfServer.warning("Non scripted personalities do not support commands",client_id=client_id)
            lollmsElfServer.close_message(client_id)
            lollmsElfServer.busy=False

            #tpe = threading.Thread(target=lollmsElfServer.start_message_generation, args=(message, message_id, client_id))
            #tpe.start()
        else:
            lollmsElfServer.error("I am busy. Come back later.", client_id=client_id)

        lollmsElfServer.busy=False

    