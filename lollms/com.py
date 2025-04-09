from ascii_colors import ASCIIColors
from lollms.types import MSG_OPERATION_TYPE, SENDER_TYPES, MSG_TYPE
from typing import Callable, Any
import socketio
from enum import Enum
from lollms.types import MSG_OPERATION_TYPE
from lollms.templating import LollmsLLMTemplate
from typing import Any, List
import asyncio
class NotificationType(Enum):
    """Notification types."""
    
    NOTIF_ERROR = 0
    """This is an error notification."""
    
    NOTIF_SUCCESS = 1
    """This is a success notification."""

    NOTIF_INFO = 2
    """This is an information notification."""

    NOTIF_WARNING = 3
    """This is a warining notification."""

class NotificationDisplayType(Enum):
    """Notification display types."""
    
    TOAST = 0
    """This is a toast."""
    
    MESSAGE_BOX = 1
    """This is a message box."""

    YESNO_MESSAGE = 2
    """This is a yes not messagebox."""

    SHOW_BLOCKING_MESSAGE = 3
    """This shows a blocking messagebox."""

    HIDE_BLOCKING_MESSAGE = 4
    """This hides a blocking messagebox."""


class LoLLMsCom:
    def __init__(self, sio:socketio.AsyncServer=None, verbose:bool=False) -> None:
        self.sio= sio
        self.verbose = verbose
        self.config = None 
        self.template:LollmsLLMTemplate = None
        self.tti = None
        self.tts = None
        self.stt = None

        self.rt_com = None

        self.model = None
        try:
            self.loop = asyncio.get_running_loop()
        except Exception as ex:
            self.loop = None
    # Helper to schedule tasks - ensures self.loop is checked
    def schedule_task(self, coro):
        if self.loop:
            # Create task schedules it, doesn't wait
            self.loop.create_task(coro)
        else:
                # This ideally shouldn't happen if setup is correct
                print(f"ERROR: Loop not available when trying to schedule {coro}")

    def InfoMessage(self, content, client_id=None, verbose:bool=None):
        self.schedule_task( self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=0, 
                client_id=client_id, 
                display_type=NotificationDisplayType.MESSAGE_BOX,
                verbose=verbose
            )
        )
    def ShowBlockingMessage(self, content, client_id=None, verbose:bool=None):
        self.schedule_task( self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=0, 
                client_id=client_id, 
                display_type=NotificationDisplayType.SHOW_BLOCKING_MESSAGE,
                verbose=verbose
            )     
        )   
        
    def HideBlockingMessage(self, client_id=None, verbose:bool=None):
        self.schedule_task( self.notify(
                "", 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=0, 
                client_id=client_id, 
                display_type=NotificationDisplayType.HIDE_BLOCKING_MESSAGE,
                verbose=verbose
            )        
        )


    async def YesNoMessage(self, content, duration:int=4, client_id=None, verbose:bool=None):
        infos={
            "wait":True,
            "result":False
        }
        @self.sio.on('yesNoRes')
        def yesnores(result):
            infos["result"] = result["yesRes"]
            infos["wait"]=False

        await self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.YESNO_MESSAGE,
                verbose=verbose
            )
        # wait
        ASCIIColors.yellow("Waiting for yes no question to be answered")
        while infos["wait"]:
            self.sio.sleep(1)
        return infos["result"]

    def close_message(self, client_id):
        pass
    
    def info(self, content, duration:int=4, client_id=None, verbose:bool=None):
        self.schedule_task( self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.TOAST,
                verbose=verbose
            )
        )
    def warning(self, content, duration:int=4, client_id=None, verbose:bool=None):
        self.schedule_task(self.notify(
                content, 
                notification_type=NotificationType.NOTIF_WARNING, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.TOAST,
                verbose=verbose
            )
        )

    def success(self, content, duration:int=4, client_id=None, verbose:bool=None):
        self.schedule_task(self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.TOAST,
                verbose=verbose
            )
        )
    def error(self, content, duration:int=4, client_id=None, verbose:bool=None):
        self.schedule_task(self.notify(
                content, 
                notification_type=NotificationType.NOTIF_ERROR, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.TOAST,
                verbose = verbose
            )
        )
    async def new_message(self, 
                            client_id, 
                            sender=None, 
                            content="",
                            parameters=None,
                            metadata=None,
                            ui=None,
                            message_type:MSG_TYPE=MSG_TYPE.MSG_TYPE_CONTENT, 
                            sender_type:SENDER_TYPES=SENDER_TYPES.SENDER_TYPES_AI,
                            open=False
                        ):
        pass

    async def start_message_generation(
        self,
        message,
        message_id,
        client_id,
        is_continue=False,
        generation_type=None,
        force_using_internet=False,
    ):
        pass
    def set_message_content(self, full_text:str, callback: Callable[[str | list | None, MSG_OPERATION_TYPE, str, Any | None], bool]=None):
        """This sends full text to front end

        Args:
            step_text (dict): The step text
            callback (callable, optional): A callable with this signature (str, MSG_TYPE) to send the text to. Defaults to None.
        """
        pass

    
    def emit_socket_io_info(self, name, data, client_id):
        pass

    async def notify(
                self, 
                content:str, 
                notification_type:NotificationType=NotificationType.NOTIF_SUCCESS, 
                duration:int=4, 
                client_id=None, 
                display_type:NotificationDisplayType=NotificationDisplayType.TOAST,
                verbose:bool|None=None
            ):
        if verbose is None:
            verbose = self.verbose

        if verbose:
            if notification_type==NotificationType.NOTIF_SUCCESS:
                ASCIIColors.success(content)
            elif notification_type==NotificationType.NOTIF_INFO:
                ASCIIColors.info(content)
            elif notification_type==NotificationType.NOTIF_WARNING:
                ASCIIColors.warning(content)
            else:
                ASCIIColors.red(content)


    async def notify_model_install(self, 
                            installation_path,
                            model_name,
                            binding_folder,
                            model_url,
                            start_time,
                            total_size,
                            downloaded_size,
                            progress,
                            speed,
                            client_id,
                            status=True,
                            error="",
                             ):
        pass