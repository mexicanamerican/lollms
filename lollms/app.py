from lollms.main_config import LOLLMSConfig
from lollms.paths import LollmsPaths
from lollms.personality import PersonalityBuilder, AIPersonality
from lollms.binding import LLMBinding, BindingBuilder, ModelBuilder
from lollms.databases.discussions_database import Message
from lollms.config import InstallOption
from lollms.helpers import ASCIIColors, trace_exception
from lollms.com import NotificationType, NotificationDisplayType, LoLLMsCom
from lollms.terminal import MainMenu
from lollms.types import MSG_OPERATION_TYPE, SENDER_TYPES
from lollms.utilities import PromptReshaper
from lollms.client_session import Client, Session
from lollms.databases.skills_database import SkillsLibrary
from lollms.tasks import TasksLibrary
from lollms.prompting import LollmsLLMTemplate, LollmsContextDetails
import importlib

from lollmsvectordb.database_elements.chunk import Chunk
from lollmsvectordb.vector_database import VectorDatabase
from typing import Callable, Any
from pathlib import Path
from datetime import datetime
from functools import partial
from socketio import AsyncServer
from typing import Tuple, List, Dict
import subprocess
import importlib
import sys, os
import platform
import gc
import yaml
import time
from lollms.utilities import run_with_current_interpreter
import socket
import json
import pipmaster as pm

import importlib.util
class LollmsApplication(LoLLMsCom):
    def __init__(
                    self, 
                    app_name:str, 
                    config:LOLLMSConfig, 
                    lollms_paths:LollmsPaths, 
                    load_binding=True, 
                    load_model=True, 
                    try_select_binding=False, 
                    try_select_model=False,
                    callback=None,
                    sio:AsyncServer=None,
                    free_mode=False
                ) -> None:
        """
        Creates a LOLLMS Application
        """
        super().__init__(sio)
        self.app_name                   = app_name
        self.config                     = config
        ASCIIColors.warning(f"Configuration fix ")
        try:
            config.personalities = [p.split(":")[0] for p in config.personalities]
            config.save_config()
        except Exception as ex:
            trace_exception(ex)

        self.lollms_paths               = lollms_paths

        # TODO : implement
        self.embedding_models           = []

        self.menu                       = MainMenu(self, callback)
        self.mounted_personalities      = []
        self.personality:AIPersonality  = None

        self.mounted_extensions         = []
        self.binding                    = None
        self.model:LLMBinding           = None
        self.long_term_memory           = None

        self.tts                        = None

        self.handle_generate_msg: Callable[[str, Dict], None]               = None
        self.generate_msg_with_internet: Callable[[str, Dict], None]        = None
        self.handle_continue_generate_msg_from: Callable[[str, Dict], None] = None
        
        # Trust store 
        self.bk_store = None
        
        # services
        self.ollama         = None
        self.vllm           = None
        self.tti = None
        self.tts = None
        self.stt = None
        self.ttm = None
        self.ttv = None
        
        self.rt_com = None
        self.is_internet_available = self.check_internet_connection()
        self.template = LollmsLLMTemplate(self.config, self.personality)

        if not free_mode:
            try:
                if config.auto_update and self.is_internet_available:
                    # Clone the repository to the target path
                    if self.lollms_paths.lollms_core_path.exists():
                        def check_lollms_core():
                            subprocess.run(["git", "-C", self.lollms_paths.lollms_core_path, "pull"]) 
                        ASCIIColors.blue("Lollms_core found in the app space.")           
                        ASCIIColors.execute_with_animation("Pulling last lollms_core", check_lollms_core)

                    def check_lollms_bindings_zoo():
                        subprocess.run(["git", "-C", self.lollms_paths.bindings_zoo_path, "pull"])
                    ASCIIColors.blue("Bindings zoo found in your personal space.")
                    ASCIIColors.execute_with_animation("Pulling last bindings zoo", check_lollms_bindings_zoo)

                    # Pull the repository if it already exists
                    def check_lollms_personalities_zoo():
                        subprocess.run(["git", "-C", self.lollms_paths.personalities_zoo_path, "pull"])            
                    ASCIIColors.blue("Personalities zoo found in your personal space.")
                    ASCIIColors.execute_with_animation("Pulling last personalities zoo", check_lollms_personalities_zoo)

                    # Pull the repository if it already exists
                    def check_lollms_models_zoo():
                        subprocess.run(["git", "-C", self.lollms_paths.models_zoo_path, "pull"])            
                    ASCIIColors.blue("Models zoo found in your personal space.")
                    ASCIIColors.execute_with_animation("Pulling last Models zoo", check_lollms_models_zoo)

                    # Pull the repository if it already exists
                    def check_lollms_function_calling_zoo():
                        subprocess.run(["git", "-C", self.lollms_paths.functions_zoo_path, "pull"])            
                    ASCIIColors.blue("Function calling zoo found in your personal space.")
                    ASCIIColors.execute_with_animation("Pulling last Function calling zoo", check_lollms_function_calling_zoo)

                    # Pull the repository if it already exists
                    def check_lollms_services_zoo():
                        subprocess.run(["git", "-C", self.lollms_paths.services_zoo_path, "pull"])            
                    ASCIIColors.blue("Services zoo found in your personal space.")
                    ASCIIColors.execute_with_animation("Pulling last services zoo", check_lollms_services_zoo)


            except Exception as ex:
                ASCIIColors.error("Couldn't pull zoos. Please contact the main dev on our discord channel and report the problem.")
                trace_exception(ex)

            if self.config.binding_name is None:
                ASCIIColors.warning(f"No binding selected")
                if try_select_binding:
                    ASCIIColors.info("Please select a valid model or install a new one from a url")
                    self.menu.select_binding()
            else:
                if load_binding:
                    try:
                        ASCIIColors.info(f">Loading binding {self.config.binding_name}. Please wait ...")
                        self.binding = self.load_binding()
                    except Exception as ex:
                        ASCIIColors.error(f"Failed to load binding.\nReturned exception: {ex}")
                        trace_exception(ex)

                    if self.binding is not None:
                        ASCIIColors.success(f"Binding {self.config.binding_name} loaded successfully.")
                        if load_model:
                            if self.config.model_name is None:
                                ASCIIColors.warning(f"No model selected")
                                if try_select_model:
                                    print("Please select a valid model")
                                    self.menu.select_model()
                                    
                            if self.config.model_name is not None:
                                try:
                                    ASCIIColors.info(f">Loading model {self.config.model_name}. Please wait ...")
                                    self.model          = self.load_model()
                                    if self.model is not None:
                                        ASCIIColors.success(f"Model {self.config.model_name} loaded successfully.")

                                except Exception as ex:
                                    ASCIIColors.error(f"Failed to load model.\nReturned exception: {ex}")
                                    trace_exception(ex)
                    else:
                        ASCIIColors.warning(f"Couldn't load binding {self.config.binding_name}.")
                
            self.mount_personalities()
            self.mount_extensions()
            
            try:
                self.load_rag_dbs()
            except Exception as ex:
                trace_exception(ex)
                
                
        self.session                    = Session(lollms_paths)
        self.skills_library             = SkillsLibrary(self.lollms_paths.personal_skills_path/(self.config.skills_lib_database_name+".sqlite"), config = self.config)
        self.tasks_library              = TasksLibrary(self)


    def load_function_call(self, fc, client):
        dr = Path(fc["dir"])
        try:
            with open(dr/"config.yaml", "r") as f:
                fc_dict = yaml.safe_load(f.read())
                # let us check static settings from fc_dict
                # Step 1: Construct the full path to the function.py module
                module_path = dr / "function.py"
                module_name = "function"  # Name for the loaded module

                # Step 2: Use importlib.util to load the module from the file path
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None:
                    raise ImportError(f"Could not load module from {module_path}")
                
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module  # Add the module to sys.modules
                spec.loader.exec_module(module)    # Execute the module

                # Step 3: Retrieve the class from the module using the class name
                class_name = fc_dict["class_name"]
                class_ = getattr(module, class_name)
                
                # Step 4: Create an instance of the class and store it in fc_dict["class"]
                fc_dict["class"] = class_(self, client)
                return fc_dict
        except Exception as ex:
            self.error("Couldn't add function call to context")
            trace_exception(ex)
        return None
    
    def execute_function(self, code, client):
        function_call=json.loads(code)
        name = function_call["function_name"]
        for fc in self.config.mounted_function_calls:
            if fc["selected"]:
                if fc["name"] == name:
                    fci = self.load_function_call(fc, client)
                    if fci:
                        output = fci["class"].execute(LollmsContextDetails(client),**function_call["function_parameters"])
                        return output
        


    def embed_function_call_in_prompt(self, original_prompt):
        """Embeds function call descriptions in the system prompt"""
        function_descriptions = [
            "You have access to these functions. Use them when needed:",
            "Format: <lollms_function_call>{JSON}</lollms_function_call>"
        ]
        
        # Get mounted functions
        mounted_functions = [
            fc for fc in self.config.mounted_function_calls 
            if fc["mounted"]
        ]
        
        for fc in mounted_functions:
            try:
                # Load function config
                fn_path = self.paths.functions_zoo_path / fc["name"]
                with open(fn_path/"config.yaml") as f:
                    config = yaml.safe_load(f)
                    
                # Build function description
                desc = [
                    f"Function: {config['name']}",
                    f"Description: {config['description']}",
                    f"Parameters: {json.dumps(config.get('parameters', {}))}",
                    f"Returns: {json.dumps(config.get('returns', {}))}",
                    f"Needs Processing: {str(config.get('needs_processing', True)).lower()}",
                    f"Examples: {', '.join(config.get('examples', []))}"
                ]
                function_descriptions.append("\n".join(desc))
                
            except Exception as e:
                print(f"Error loading function {fc['name']}: {e}")

        return original_prompt + "\n\n" + "\n\n".join(function_descriptions)

    def detect_function_calls(self, text):
        """Detects and parses function calls in AI output"""
        import re
        import json
        
        pattern = r'<lollms_function_call>(.*?)</lollms_function_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        valid_calls = []
        
        for match in matches:
            try:
                call_data = json.loads(match.strip())
                
                # Validate required fields
                if not all(k in call_data for k in ["function_name", "parameters"]):
                    continue
                    
                # Check if function is mounted
                is_mounted = any(
                    fc["name"] == call_data["function_name"] and fc["mounted"]
                    for fc in self.config.mounted_function_calls
                )
                
                if not is_mounted:
                    continue
                    
                # Set default needs_processing if missing
                if "needs_processing" not in call_data:
                    call_data["needs_processing"] = True
                    
                valid_calls.append({
                    "function_name": call_data["function_name"],
                    "parameters": call_data["parameters"],
                    "needs_processing": call_data["needs_processing"]
                })
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error parsing function call: {e}")
                continue
                
        return valid_calls

    @staticmethod
    def check_internet_connection():
        global is_internet_available
        try:
            # Attempt to connect to a reliable server (in this case, Google's DNS)
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            is_internet_available = True
            return True
        except OSError:
            is_internet_available = False
            return False


    def backup_trust_store(self):
        self.bk_store = None
        if 'REQUESTS_CA_BUNDLE' in os.environ:
            self.bk_store = os.environ['REQUESTS_CA_BUNDLE']
            del os.environ['REQUESTS_CA_BUNDLE']

    def restore_trust_store(self):
        if self.bk_store is not None:
            os.environ['REQUESTS_CA_BUNDLE'] = self.bk_store

    def model_path_to_binding_model(self, model_path:str):
        parts = model_path.strip().split("::")
        if len(parts)<2:
            raise Exception("Model path is not in the format binding:model_name!")
        binding = parts[0]
        model_name = parts[1]
        return binding, model_name
      
    def select_model(self, binding_name, model_name, destroy_previous_model=True):
        self.config["binding_name"] = binding_name
        self.config["model_name"] = model_name
        print(f"New binding selected : {binding_name}")

        try:
            if self.binding and destroy_previous_model:
                self.binding.destroy_model()
            self.binding = None
            self.model = None
            for per in self.mounted_personalities:
                if per is not None:
                    per.model = None
            gc.collect()
            self.binding = BindingBuilder().build_binding(self.config, self.lollms_paths, InstallOption.INSTALL_IF_NECESSARY, lollmsCom=self)
            self.config["model_name"] = model_name
            self.model = self.binding.build_model()
            for per in self.mounted_personalities:
                if per is not None:
                    per.model = self.model
            self.config.save_config()
            ASCIIColors.green("Binding loaded successfully")
            return True
        except Exception as ex:
            ASCIIColors.error(f"Couldn't build binding: [{ex}]")
            trace_exception(ex)
            return False
        

    def set_active_model(self, model):
        print(f"New model active : {model.model_name}")
        self.model = model
        self.binding = model
        self.personality.model = model
        for per in self.mounted_personalities:
            if per is not None:
                per.model = self.model
        self.config["binding_name"] = model.binding_folder_name
        self.config["model_name"] = model.model_name

                
    def add_discussion_to_skills_library(self, client: Client):
        messages = client.discussion.get_messages()

        # Extract relevant information from messages
        def cb(str, MSG_TYPE_=MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_SET_CONTENT, dict=None, list=None):
            if MSG_TYPE_!=MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK:
                self.ShowBlockingMessage(f"Learning\n{str}")
        bk_cb = self.tasks_library.callback
        self.tasks_library.callback = cb
        content = self._extract_content(messages, cb)
        self.tasks_library.callback = bk_cb

        # Generate title
        title_prompt =  f"{self.separator_template}".join([
            f"{self.system_full_header}Generate a concise and descriptive title and category for the following content:",
            content
            ])
        template =  f"{self.separator_template}".join([
            "{",
            '   "title":"here you put the title"',
            '   "category":"here you put the category"',
            "}"])
        language = "json"
        title_category_json = json.loads(self._generate_code(title_prompt, template, language))
        title = title_category_json["title"]
        category = title_category_json["category"]

        # Add entry to skills library
        self.skills_library.add_entry(1, category, title, content)
        return category, title, content

    def _extract_content(self, messages:List[Message], callback = None):      
        message_content = ""

        for message in messages:
            rank = message.rank
            sender = message.sender
            text = message.content
            message_content += f"Rank {rank} - {sender}: {text}\n"

        return self.tasks_library.summarize_text(
            message_content,
            "\n".join([
                "Find out important information from the discussion and report them.",
                "Format the output as sections if applicable:",
                "Global context: Explain in a sentense or two the subject of the discussion",
                "Interesting things (if applicable): If you find interesting information or something that was discovered or built in this discussion, list it here with enough details to be reproducible just by reading this text.",
                "Code snippet (if applicable): If there are important code snippets, write them here in a markdown code tag.",
                "Make the output easy to understand.",
                "The objective is not to talk about the discussion but to store the important information for future usage. Do not report useless information.",
                "Do not describe the discussion and focuse more on reporting the most important information from the discussion."
            ]),
            doc_name="discussion",
            callback=callback)
        

    def _generate_text(self, prompt):
        max_tokens = min(self.config.ctx_size - self.model.get_nb_tokens(prompt),self.config.max_n_predict if self.config.max_n_predict else self.config.ctx_size- self.model.get_nb_tokens(prompt))
        generated_text = self.model.generate(prompt, max_tokens)
        return generated_text.strip()
    
    def _generate_code(self, prompt, template, language):
        max_tokens = min(self.config.ctx_size - self.model.get_nb_tokens(prompt),self.config.max_n_predict if self.config.max_n_predict else self.config.ctx_size- self.model.get_nb_tokens(prompt))
        generated_code = self.personality.generate_code(prompt, self.personality.image_files, template, language, max_size= max_tokens)
        return generated_code

    def get_uploads_path(self, client_id):
        return self.lollms_paths.personal_uploads_path
    
    def load_rag_dbs(self):
        ASCIIColors.info("Loading RAG datalakes")
        self.active_datalakes = []
        for rag_db in self.config.datalakes:
            if rag_db['mounted']:
                if rag_db['type']=='lollmsvectordb':
                    try:                    
                        from lollmsvectordb import VectorDatabase
                        from lollmsvectordb.text_document_loader import TextDocumentsLoader
                        from lollmsvectordb.lollms_tokenizers.tiktoken_tokenizer import TikTokenTokenizer

                        # Vectorizer selection
                        if self.config.rag_vectorizer == "semantic":
                            from lollmsvectordb.lollms_vectorizers.semantic_vectorizer import SemanticVectorizer
                            vectorizer = SemanticVectorizer(self.config.rag_vectorizer_model)
                        elif self.config.rag_vectorizer == "tfidf":
                            from lollmsvectordb.lollms_vectorizers.tfidf_vectorizer import TFIDFVectorizer
                            vectorizer = TFIDFVectorizer()
                        elif self.config.rag_vectorizer == "openai":
                            from lollmsvectordb.lollms_vectorizers.openai_vectorizer import OpenAIVectorizer
                            vectorizer = OpenAIVectorizer(
                                self.config.rag_vectorizer_model,
                                self.config.rag_vectorizer_openai_key
                            )
                        elif self.config.rag_vectorizer == "ollama":
                            from lollmsvectordb.lollms_vectorizers.ollama_vectorizer import OllamaVectorizer
                            vectorizer = OllamaVectorizer(
                                self.config.rag_vectorizer_model,
                                self.config.rag_service_url
                            )

                        # Create database path and initialize VectorDatabase
                        db_path = Path(rag_db['path']) / f"{rag_db['alias']}.sqlite"
                        vdb = VectorDatabase(
                            db_path,
                            vectorizer,
                            None if self.config.rag_vectorizer == "semantic" else self.model if self.model else TikTokenTokenizer(),
                            n_neighbors=self.config.rag_n_chunks
                        )       

                        # Add to active databases
                        self.active_datalakes.append(
                            rag_db | {"binding": vdb}
                        )

                    except Exception as ex:
                        trace_exception(ex)
                        ASCIIColors.error(f"Couldn't load {db_path} consider revectorizing it")
                elif rag_db['type']=='lightrag':
                    from lollmsvectordb.database_clients.lightrag_client import LollmsLightRagConnector
                    lr = LollmsLightRagConnector(rag_db['url'], rag_db['key'])
                    self.active_datalakes.append(
                            rag_db | {"binding": lr}
                    )
    def load_service_from_folder(self, folder_path, target_name):
        # Convert folder_path to a Path object
        folder_path = Path(folder_path)

        # List all folders in the given directory
        folders = [f for f in folder_path.iterdir() if f.is_dir()]

        # Check if the target_name matches any folder name
        target_folder = folder_path / target_name
        if target_folder in folders:
            # Load the config.yaml file
            config_path = target_folder / "config.yaml"
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Extract the class_name from the config
            class_name = config.get('class_name')
            if not class_name:
                raise ValueError(f"class_name not found in {config_path}")

            # Load the Python file
            python_file_path = target_folder / f"service.py"
            spec = importlib.util.spec_from_file_location(target_name, python_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Import the class and instantiate it
            class_ = getattr(module, class_name)
            instance = class_(self)  # Pass the config as a parameter to the constructor

            return instance
        else:
            ASCIIColors.error(f"No folder named {target_name} found in {folder_path}")

    def start_servers(self):
        ASCIIColors.yellow("* - * - * - Starting services - * - * - *")
        def start_local_services(*args, **kwargs):
            for rag_server in self.config.rag_local_services:
                try:
                    # - alias: datalake
                    #     key: ''
                    #     path: ''
                    #     start_at_startup: false
                    #     type: lightrag
                    #     url: http://localhost:9621/

                    if rag_server["start_at_startup"]:
                        if rag_server["type"]=="lightrag":
                            try:
                                self.ShowBlockingMessage("Installing Lightrag\nPlease wait...")
                                # Define the path to the apps folder
                                if not pm.is_installed("lightrag-hku"):
                                    apps_folder = self.lollms_paths.personal_user_infos_path / "apps"
                                    apps_folder.mkdir(parents=True, exist_ok=True)  # Ensure the apps folder exists
                                    # Define the path to clone the repository
                                    clone_path = apps_folder / "LightRAG"
                                    
                                    # Clone the repository if it doesn't already exist
                                    if not clone_path.exists():
                                        subprocess.run(["git", "clone", "https://github.com/ParisNeo/LightRAG.git", str(clone_path)])
                                        print(f"Repository cloned to: {clone_path}")
                                    else:
                                        print(f"Repository already exists at: {clone_path}")
                                    
                                    # Install the package in editable mode with extras
                                    subprocess.run([sys.executable, "-m", "pip", "install", "-e", f"{str(clone_path)}[api,tools]"])                                    
                                subprocess.Popen(
                                ["lightrag-server", "--llm-binding", "lollms", "--embedding-binding", "lollms", "--input-dir", rag_server["input_path"], "--working-dir", rag_server["working_path"]],
                                text=True,
                                stdout=None, # This will make the output go directly to console
                                stderr=None  # This will make the errors go directly to console
                                )
                                self.HideBlockingMessage()
                            except Exception as ex:
                                self.HideBlockingMessage()
                                trace_exception(ex)
                except Exception as ex:
                    trace_exception(ex)
                    self.warning(f"Couldn't start lightrag")

        ASCIIColors.execute_with_animation("Loading RAG servers", start_local_services,ASCIIColors.color_blue)
        
        tts_services = []
        stt_services = []
        def start_ttt(*args, **kwargs):
            if self.config.enable_ollama_service:
                try:
                    from lollms.services.ttt.ollama.lollms_ollama import Service
                    self.ollama = Service(self, base_url=self.config.ollama_base_url)
                    tts_services.append("ollama")

                except Exception as ex:
                    trace_exception(ex)
                    self.warning(f"Couldn't load Ollama")

            if self.config.enable_vllm_service:
                try:
                    from lollms.services.ttt.vllm.lollms_vllm import Service
                    self.vllm = Service(self, base_url=self.config.vllm_url)
                    tts_services.append("vllm")
                except Exception as ex:
                    trace_exception(ex)
                    self.warning(f"Couldn't load vllm")
        ASCIIColors.execute_with_animation("Loading TTT services", start_ttt,ASCIIColors.color_blue)

        def start_stt(*args, **kwargs):
            if self.config.whisper_activate or self.config.active_stt_service == "whisper":
                try:
                    from lollms.services.stt.whisper.lollms_whisper import LollmsWhisper
                    self.whisper = LollmsWhisper(self)
                    stt_services.append("whisper")
                except Exception as ex:
                    trace_exception(ex)
            if self.config.active_stt_service == "openai_whisper":
                from lollms.services.stt.openai_whisper.lollms_openai_whisper import LollmsOpenAIWhisper
                self.stt = LollmsOpenAIWhisper(self)
            elif self.config.active_stt_service == "whisper":
                from lollms.services.stt.whisper.lollms_whisper import LollmsWhisper
                self.stt = LollmsWhisper(self)

        ASCIIColors.execute_with_animation("Loading STT services", start_stt, ASCIIColors.color_blue)

        def start_tts(*args, **kwargs):
            if self.config.active_tts_service == "xtts":
                ASCIIColors.yellow("Loading XTTS")
                try:
                    from lollms.services.tts.xtts.lollms_xtts import LollmsXTTS

                    self.tts = LollmsXTTS(
                                            self
                                        )
                except Exception as ex:
                    trace_exception(ex)
                    self.warning(f"Couldn't load XTTS")
            if self.config.active_tts_service == "eleven_labs_tts":
                from lollms.services.tts.eleven_labs_tts.lollms_eleven_labs_tts import LollmsElevenLabsTTS
                self.tts = LollmsElevenLabsTTS(self)
            elif self.config.active_tts_service == "openai_tts":
                from lollms.services.tts.open_ai_tts.lollms_openai_tts import LollmsOpenAITTS
                self.tts = LollmsOpenAITTS(self)
            elif self.config.active_tts_service == "fish_tts":
                from lollms.services.tts.fish.lollms_fish_tts import LollmsFishAudioTTS
                self.tts = LollmsFishAudioTTS(self)

        ASCIIColors.execute_with_animation("Loading TTS services", start_tts, ASCIIColors.color_blue)

        def start_tti(*args, **kwargs):
            self.tti = self.load_service_from_folder(self.lollms_paths.services_zoo_path/"tti", self.config.active_tti_service)
        ASCIIColors.execute_with_animation("Loading loacal TTI services", start_tti, ASCIIColors.color_blue)

        def start_ttv(*args, **kwargs):
            self.ttv = self.load_service_from_folder(self.lollms_paths.services_zoo_path/"ttv", self.config.active_ttv_service)


        ASCIIColors.execute_with_animation("Loading loacal TTV services", start_ttv, ASCIIColors.color_blue)
        print("OK")



    def verify_servers(self, reload_all=False):
        ASCIIColors.yellow("* - * - * - Verifying services - * - * - *")

        try:
            ASCIIColors.blue("Loading active local TTT services")
            
            if self.config.enable_ollama_service and self.ollama is None:
                try:
                    from lollms.services.ttt.ollama.lollms_ollama import Service
                    self.ollama = Service(self, base_url=self.config.ollama_base_url)
                except Exception as ex:
                    trace_exception(ex)
                    self.warning(f"Couldn't load Ollama")

            if self.config.enable_vllm_service and self.vllm is None:
                try:
                    from lollms.services.ttt.vllm.lollms_vllm import Service
                    self.vllm = Service(self, base_url=self.config.vllm_url)
                except Exception as ex:
                    trace_exception(ex)
                    self.warning(f"Couldn't load vllm")

            ASCIIColors.blue("Loading local STT services")

            if self.config.whisper_activate and self.whisper is None:
                try:
                    from lollms.services.stt.whisper.lollms_whisper import LollmsWhisper
                    self.whisper = LollmsWhisper(self)
                except Exception as ex:
                    trace_exception(ex)
                    
            ASCIIColors.blue("Loading loacal TTS services")
            if self.config.active_tts_service == "xtts" and (self.tts is None or self.tts.name!="xtts"):
                ASCIIColors.yellow("Loading XTTS")
                try:
                    from lollms.services.tts.xtts.lollms_xtts import LollmsXTTS
                    voice=self.config.xtts_current_voice
                    if voice!="main_voice":
                        voices_folder = self.lollms_paths.custom_voices_path
                    else:
                        voices_folder = Path(__file__).parent.parent.parent/"services/xtts/voices"

                    self.tts = LollmsXTTS(
                                            self
                                        )
                except Exception as ex:
                    trace_exception(ex)
                    self.warning(f"Couldn't load XTTS")

            def start_tti(*args, **kwargs):
                self.tti = self.load_service_from_folder(self.lollms_paths.services_zoo_path/"tti", self.config.active_tti_service)
            ASCIIColors.execute_with_animation("Loading loacal TTI services", start_tti, ASCIIColors.color_blue)


            ASCIIColors.blue("Activating TTS service")
            if self.config.active_tts_service == "eleven_labs_tts":
                from lollms.services.tts.eleven_labs_tts.lollms_eleven_labs_tts import LollmsElevenLabsTTS
                self.tts = LollmsElevenLabsTTS(self)
            elif self.config.active_tts_service == "openai_tts" and (self.tts is None or self.tts.name!="openai_tts"):
                from lollms.services.tts.open_ai_tts.lollms_openai_tts import LollmsOpenAITTS
                self.tts = LollmsOpenAITTS(self)
            elif self.config.active_tts_service == "fish_tts":
                from lollms.services.tts.fish.lollms_fish_tts import LollmsFishAudioTTS
                self.tts = LollmsFishAudioTTS(self)

            ASCIIColors.blue("Activating STT service")
            if self.config.active_stt_service == "openai_whisper" and (self.tts is None or self.tts.name!="openai_whisper"):
                from lollms.services.stt.openai_whisper.lollms_openai_whisper import LollmsOpenAIWhisper
                self.stt = LollmsOpenAIWhisper(self)
            elif self.config.active_stt_service == "whisper" and (self.tts is None or  self.tts.name!="whisper") :
                from lollms.services.stt.whisper.lollms_whisper import LollmsWhisper
                self.stt = LollmsWhisper(self)


            def start_ttv(*args, **kwargs):
                self.ttv = self.load_service_from_folder(self.lollms_paths.services_zoo_path/"ttv", self.config.active_ttv_service)


            ASCIIColors.execute_with_animation("Loading loacal TTV services", start_ttv, ASCIIColors.color_blue)
            print("OK")



        except Exception as ex:
            trace_exception(ex)
            

    
    def process_data(
                        self, 
                        chunk:str, 
                        message_type,
                        parameters:dict=None, 
                        metadata:list=None, 
                        personality=None
                    ):
        
        pass

    def default_callback(self, chunk, type, generation_infos:dict):
        if generation_infos["nb_received_tokens"]==0:
            self.start_time = datetime.now()
        dt =(datetime.now() - self.start_time).seconds
        if dt==0:
            dt=1
        spd = generation_infos["nb_received_tokens"]/dt
        ASCIIColors.green(f"Received {generation_infos['nb_received_tokens']} tokens (speed: {spd:.2f}t/s)              ",end="\r",flush=True) 
        sys.stdout = sys.__stdout__
        sys.stdout.flush()
        if chunk:
            generation_infos["generated_text"] += chunk
        antiprompt = self.personality.detect_antiprompt(generation_infos["generated_text"])
        if antiprompt:
            ASCIIColors.warning(f"\n{antiprompt} detected. Stopping generation")
            generation_infos["generated_text"] = self.remove_text_from_string(generation_infos["generated_text"],antiprompt)
            return False
        else:
            generation_infos["nb_received_tokens"] += 1
            generation_infos["first_chunk"]=False
            # if stop generation is detected then stop
            if not self.cancel_gen:
                return True
            else:
                self.cancel_gen = False
                ASCIIColors.warning("Generation canceled")
                return False
   
    def remove_text_from_string(self, string, text_to_find):
        """
        Removes everything from the first occurrence of the specified text in the string (case-insensitive).

        Parameters:
        string (str): The original string.
        text_to_find (str): The text to find in the string.

        Returns:
        str: The updated string.
        """
        index = string.lower().find(text_to_find.lower())

        if index != -1:
            string = string[:index]

        return string

    def load_binding(self):
        try:
            binding = BindingBuilder().build_binding(self.config, self.lollms_paths, lollmsCom=self)
            return binding    
        except Exception as ex:
            self.error("Couldn't load binding")
            self.info("Trying to reinstall binding")
            trace_exception(ex)
            try:
                binding = BindingBuilder().build_binding(self.config, self.lollms_paths,installation_option=InstallOption.FORCE_INSTALL, lollmsCom=self)
            except Exception as ex:
                self.error("Couldn't reinstall binding")
                trace_exception(ex)
            return None    

    
    def load_model(self):
        try:
            model = ModelBuilder(self.binding).get_model()
            for personality in self.mounted_personalities:
                if personality is not None:
                    personality.model = model
        except Exception as ex:
            self.error("Couldn't load model.")
            ASCIIColors.error(f"Couldn't load model. Please verify your configuration file at {self.lollms_paths.personal_configuration_path} or use the next menu to select a valid model")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            trace_exception(ex)
            ASCIIColors.error(f"{self.config.get_model_path_infos()}")
            print("Please select a valid model or install a new one from a url")
            model = None

        return model



    def mount_personality(self, id:int, callback=None):
        try:
            personality = PersonalityBuilder(self.lollms_paths, self.config, self.model, self, callback=callback).build_personality(id)
            if personality.model is not None:
                try:
                    self.cond_tk = personality.model.tokenize(personality.personality_conditioning)
                    self.n_cond_tk = len(self.cond_tk)
                    ASCIIColors.success(f"Personality  {personality.name} mounted successfully")
                except:
                    self.cond_tk = []      
                    self.n_cond_tk = 0      
            else:
                ASCIIColors.success(f"Personality  {personality.name} mounted successfully but no model is selected")
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load personality. Please verify your configuration file at {self.lollms_paths.personal_configuration_path} or use the next menu to select a valid personality")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            trace_exception(ex)
            ASCIIColors.error(f"{self.config.get_personality_path_infos()}")
            if id == self.config.active_personality_id:
                self.config.active_personality_id=len(self.config.personalities)-1
            personality = None
        
        self.mounted_personalities.append(personality)
        return personality
    
    def mount_personalities(self, callback = None):
        self.mounted_personalities = []
        to_remove = []
        for i in range(len(self.config["personalities"])):
            p = self.mount_personality(i, callback = None)
            if p is None:
                to_remove.append(i)
        to_remove.sort(reverse=True)
        for i in to_remove:
            self.unmount_personality(i)

        if self.config.active_personality_id>=0 and self.config.active_personality_id<len(self.mounted_personalities):
            self.personality = self.mounted_personalities[self.config.active_personality_id]
        else:
            self.config["personalities"].insert(0, "generic/lollms")
            self.mount_personality(0, callback = None)
            self.config.active_personality_id = 0
            self.personality = self.mounted_personalities[self.config.active_personality_id]

    def mount_extensions(self, callback = None):
        self.mounted_extensions = []
        to_remove = []
        for i in range(len(self.config["extensions"])):
            p = self.mount_extension(i, callback = None)
            if p is None:
                to_remove.append(i)
        to_remove.sort(reverse=True)
        for i in to_remove:
            self.unmount_extension(i)


    def set_personalities_callbacks(self, callback: Callable[[str, int, dict], bool]=None):
        for personality in self.mount_personalities:
            personality.setCallback(callback)

    def unmount_extension(self, id:int)->bool:
        if id<len(self.config.extensions):
            del self.config.extensions[id]
            if id>=0 and id<len(self.mounted_extensions):
                del self.mounted_extensions[id]
            self.config.save_config()
            return True
        else:
            return False

            
    def unmount_personality(self, id:int)->bool:
        if id<len(self.config.personalities):
            del self.config.personalities[id]
            del self.mounted_personalities[id]
            if self.config.active_personality_id>=id:
                self.config.active_personality_id-=1

            self.config.save_config()
            return True
        else:
            return False


    def select_personality(self, id:int):
        if id<len(self.config.personalities):
            self.config.active_personality_id = id
            self.personality = self.mounted_personalities[id]
            self.config.save_config()
            return True
        else:
            return False


    def load_personality(self, callback=None):
        try:
            personality = PersonalityBuilder(self.lollms_paths, self.config, self.model, self, callback=callback).build_personality()
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load personality. Please verify your configuration file at {self.configuration_path} or use the next menu to select a valid personality")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            ASCIIColors.error(f"{self.config.get_personality_path_infos()}")
            print("Please select a valid model or install a new one from a url")
            personality = None
        return personality

    @staticmethod   
    def reset_paths(lollms_paths:LollmsPaths):
        lollms_paths.resetPaths()

    @staticmethod   
    def reset_all_installs(lollms_paths:LollmsPaths):
        ASCIIColors.info("Removeing all configuration files to force reinstall")
        ASCIIColors.info(f"Searching files from {lollms_paths.personal_configuration_path}")
        for file_path in lollms_paths.personal_configuration_path.iterdir():
            if file_path.name!=f"{lollms_paths.tool_prefix}local_config.yaml" and file_path.suffix.lower()==".yaml":
                file_path.unlink()
                ASCIIColors.info(f"Deleted file: {file_path}")


    #languages:
    def get_personality_languages(self):
        languages = [self.personality.default_language]
        persona_language_path = self.lollms_paths.personalities_zoo_path/self.personality.category/self.personality.name.replace(" ","_")/"languages"
        for language_file in persona_language_path.glob("*.yaml"):
            language_code = language_file.stem
            languages.append(language_code)
        # Construire le chemin vers le dossier contenant les fichiers de langue pour la personnalité actuelle
        languages_dir = self.lollms_paths.personal_configuration_path / "personalities" / self.personality.name
        if self.personality.language:
            default_language = self.personality.language.lower().strip().split()[0]
        else:
            default_language = "english"
        # Vérifier si le dossier existe
        languages_dir.mkdir(parents=True, exist_ok=True)
        
        # Itérer sur chaque fichier YAML dans le dossier
        for language_file in languages_dir.glob("languages_*.yaml"):
            # Improved extraction of the language code to handle names with underscores
            parts = language_file.stem.split("_")
            if len(parts) > 2:
                language_code = "_".join(parts[1:])  # Rejoin all parts after "languages"
            else:
                language_code = parts[-1]
            
            if language_code != default_language and language_code not in languages:
                languages.append(language_code)
        
        return languages



    def set_personality_language(self, language:str):
        if language is None or  language == "":
            return False
        language = language.lower().strip().split()[0]
        self.personality.set_language(language)

        self.config.current_language=language
        self.config.save_config()
        return True

    def del_personality_language(self, language:str):
        if language is None or  language == "":
            return False
        
        language = language.lower().strip().split()[0]
        default_language = self.personality.language.lower().strip().split()[0]
        if language == default_language:
            return False # Can't remove the default language
                
        language_path = self.lollms_paths.personal_configuration_path/"personalities"/self.personality.name/f"languages_{language}.yaml"
        if language_path.exists():
            try:
                language_path.unlink()
            except Exception as ex:
                return False
            if self.config.current_language==language:
                self.config.current_language="english"
                self.config.save_config()
        return True

    def recover_discussion(self,client_id, message_index=-1):
        messages = self.session.get_client(client_id).discussion.get_messages()
        discussion=""
        for msg in messages[:-1]:
            if message_index!=-1 and msg>message_index:
                break
            discussion += "\n" + self.config.discussion_prompt_separator + msg.sender + ": " + msg.content.strip()
        return discussion
    # -------------------------------------- Prompt preparing
    def prepare_query(self, client_id: str, message_id: int = -1, is_continue: bool = False, n_tokens: int = 0, generation_type = None, force_using_internet=False, previous_chunk="") -> LollmsContextDetails:
        """
        Prepares the query for the model.

        Args:
            client_id (str): The client ID.
            message_id (int): The message ID. Default is -1.
            is_continue (bool): Whether the query is a continuation. Default is False.
            n_tokens (int): The number of tokens. Default is 0.

        Returns:
            Tuple[str, str, List[str]]: The prepared query, original message content, and tokenized query.
        """
        skills_detials=[]
        skills = []
        documentation_entries = []

        if self.personality.callback is None:
            self.personality.callback = partial(self.process_data, client_id=client_id)
        # Get the list of messages
        client = self.session.get_client(client_id)
        discussion = client.discussion
        messages = discussion.get_messages()

        # Find the index of the message with the specified message_id
        message_index = -1
        for i, message in enumerate(messages):
            if message.id == message_id:
                message_index = i
                break
        
        # Define current message
        current_message = messages[message_index]

        # Build the conditionning text block
        default_language = self.personality.language.lower().strip().split()[0]
        current_language = self.config.current_language.lower().strip().split()[0]

        if current_language and  current_language!= self.personality.language:
            language_path = self.lollms_paths.personal_configuration_path/"personalities"/self.personality.name/f"languages_{current_language}.yaml"
            if not language_path.exists():
                self.info(f"This is the first time this personality speaks {current_language}\nLollms is reconditionning the persona in that language.\nThis will be done just once. Next time, the personality will speak {current_language} out of the box")
                language_path.parent.mkdir(exist_ok=True, parents=True)
                # Translating
                conditionning = self.tasks_library.translate_conditionning(self.personality._personality_conditioning, self.personality.language, current_language)
                welcome_message = self.tasks_library.translate_message(self.personality.welcome_message, self.personality.language, current_language)
                with open(language_path,"w",encoding="utf-8", errors="ignore") as f:
                    yaml.safe_dump({"personality_conditioning":conditionning,"welcome_message":welcome_message}, f)
            else:
                with open(language_path,"r",encoding="utf-8", errors="ignore") as f:
                    language_pack = yaml.safe_load(f)
                    conditionning = language_pack.get("personality_conditioning", language_pack.get("conditionning", self.personality.personality_conditioning))
        else:
            conditionning = self.personality._personality_conditioning

        if len(conditionning)>0:
            if type(conditionning) is list:
                conditionning = "\n".join(conditionning)
            conditionning =  self.system_full_header + conditionning + ("" if conditionning[-1]==self.separator_template else self.separator_template)

        # Check if there are document files to add to the prompt
        internet_search_results = ""
        internet_search_infos = []
        documentation = ""


        # boosting information
        if self.config.positive_boost:
            positive_boost=f"{self.system_custom_header('important information')}"+self.config.positive_boost+"\n"
            n_positive_boost = len(self.model.tokenize(positive_boost))
        else:
            positive_boost=""
            n_positive_boost = 0

        if self.config.negative_boost:
            negative_boost=f"{self.system_custom_header('important information')}"+self.config.negative_boost+"\n"
            n_negative_boost = len(self.model.tokenize(negative_boost))
        else:
            negative_boost=""
            n_negative_boost = 0

        if self.config.fun_mode:
            fun_mode=f"""{self.system_custom_header('important information')} 
Fun mode activated. In this mode you must answer in a funny playful way. Do not be serious in your answers. Each answer needs to make the user laugh.\n"
"""
            n_fun_mode = len(self.model.tokenize(positive_boost))
        else:
            fun_mode=""
            n_fun_mode = 0

        if self.config.think_first_mode:
            think_first_mode=f"""{self.system_custom_header('important information')} 
{self.config.thinking_prompt}
"""
            n_think_first_mode = len(self.model.tokenize(positive_boost))
        else:
            think_first_mode=""
            n_think_first_mode = 0

        discussion = None
        if generation_type != "simple_question":

            # Standard RAG
            if not self.personality.ignore_discussion_documents_rag:
                if self.personality.persona_data_vectorizer or len(self.active_datalakes) > 0 or ((len(client.discussion.text_files) > 0) and client.discussion.vectorizer is not None) or self.config.activate_skills_lib:
                    #Prepare query

                    # Recover or initialize discussion
                    if discussion is None:
                        discussion = self.recover_discussion(client_id)

                    # Build documentation if empty
                    if documentation == "":
                        documentation = f"{self.separator_template}".join([
                            f"{self.system_custom_header('important information')}",
                            "Utilize Documentation Data: Always refer to the provided documentation to answer user questions accurately.",
                            "Absence of Information: If the required information is not available in the documentation, inform the user that the requested information is not present in the documentation section.",
                            "Strict Adherence to Documentation: It is strictly prohibited to provide answers without concrete evidence from the documentation.",
                            "Cite Your Sources: After providing an answer, include the full path to the document where the information was found.",
                            f"{self.system_custom_header('Documentation')}"
                        ])
                        documentation += f"{self.separator_template}"

                    # Process query
                    if self.config.rag_build_keys_words:
                        self.personality.step_start("Building vector store query")
                        prompt = f"""{self.system_full_header} You are a prompt to query converter assistant. Read the discussion and rewrite the last user prompt as a self sufficient prompt containing all neeeded information.\n
Do not answer the prompt. Do not add explanations.
{self.separator_template}
--- discussion ---
{self.system_custom_header('discussion')}'\n{discussion[-2048:]}
---
Answer directly with the reformulation of the last prompt.
{self.ai_custom_header('assistant')}"""
                        query = self.personality.fast_gen(
                            prompt,
                            max_generation_size=256,
                            show_progress=True,
                            callback=self.personality.sink
                        )
                        query = self.personality.remove_thinking_blocks(query)
                        self.personality.step_end("Building vector store query")
                        self.personality.step(f"Query: {query}")
                    else:
                        query = current_message.content

                    # Inform the user    
                    self.personality.step_start("Querying the RAG datalake")

                    # RAGs
                    if len(self.active_datalakes) > 0:
                        recovered_ids=[[] for _ in range(len(self.active_datalakes))]
                        for i,db in enumerate(self.active_datalakes):
                            if db['mounted']:
                                try:
                                    if db["type"]=="lollmsvectordb":
                                        from lollmsvectordb.vector_database import VectorDatabase
                                        binding:VectorDatabase = db["binding"]

                                        r=binding.search(query, self.config.rag_n_chunks, recovered_ids[i])
                                        recovered_ids[i]+=[rg.chunk_id for rg in r]
                                        if self.config.rag_activate_multi_hops:
                                            r = [rg for rg in r if self.personality.verify_rag_entry(query, rg.text)]
                                        documentation += "\n".join(["## chunk" + research_result.text  for research_result in r])+"\n"
                                    elif db["type"]=="lightrag":
                                        try:
                                            from lollmsvectordb.database_clients.lightrag_client import LollmsLightRagConnector
                                            lc:LollmsLightRagConnector = db["binding"]
                                            documentation += lc.query(query)
                                                
                                        except Exception as ex:
                                            trace_exception(ex)
                                except Exception as ex:
                                    trace_exception(ex)
                                    self.personality.error(f"Couldn't recover information from Datalake {db['alias']}")

                    if self.personality.persona_data_vectorizer:
                        chunks:List[Chunk] = self.personality.persona_data_vectorizer.search(query, int(self.config.rag_n_chunks))
                        for chunk in chunks:
                            if self.config.rag_put_chunk_informations_into_context:
                                documentation += f"{self.system_custom_header('document chunk')}\n## document title: {chunk.doc.title}\n## chunk content:\n{chunk.text}\n"
                            else:
                                documentation += f"{self.system_custom_header('chunk')}\n{chunk.text}\n"

                    if (len(client.discussion.text_files) > 0) and client.discussion.vectorizer is not None:
                        chunks:List[Chunk] = client.discussion.vectorizer.search(query, int(self.config.rag_n_chunks))
                        for chunk in chunks:
                            if self.config.rag_put_chunk_informations_into_context:
                                documentation += f"{self.system_custom_header('document chunk')}\n## document title: {chunk.doc.title}\n## chunk content:\n{chunk.text}\n"
                            else:
                                documentation += f"{self.start_header_id_template}chunk{self.end_header_id_template}\n{chunk.text}\n"                    
                    # Check if there is discussion knowledge to add to the prompt
                    if self.config.activate_skills_lib:
                        try:
                            # skills = self.skills_library.query_entry(query)
                            self.personality.step_start("Adding skills")
                            if self.config.debug:
                                ASCIIColors.info(f"Query : {query}")
                            skill_titles, skills, similarities = self.skills_library.query_vector_db(query, top_k=3, min_similarity=self.config.rag_min_correspondance)#query_entry_fts(query)
                            skills_detials=[{"title": title, "content":content, "similarity":similarity} for title, content, similarity in zip(skill_titles, skills, similarities)]

                            if len(skills)>0:
                                if documentation=="":
                                    documentation=f"{self.system_custom_header('skills library knowledges')}\n"
                                for i,skill in enumerate(skills_detials):
                                    documentation += "---\n"+ self.system_custom_header(f"knowledge {i}") +f"\ntitle:\n{skill['title']}\ncontent:\n{skill['content']}\n---\n"
                            self.personality.step_end("Adding skills")
                        except Exception as ex:
                            trace_exception(ex)
                            self.warning("Couldn't add long term memory information to the context. Please verify the vector database")        # Add information about the user
                            self.personality.step_end("Adding skills")

                    # Inform the user    
                    self.personality.step_end("Querying the RAG datalake")

                    documentation += f"{self.separator_template}{self.system_custom_header('important information')}Use the documentation data to answer the user questions. If the data is not present in the documentation, please tell the user that the information he is asking for does not exist in the documentation section. It is strictly forbidden to give the user an answer without having actual proof from the documentation.\n"


            # Internet
            if self.config.activate_internet_search or force_using_internet or generation_type == "full_context_with_internet":
                if discussion is None:
                    discussion = self.recover_discussion(client_id)
                if self.config.internet_activate_search_decision:
                    self.personality.step_start(f"Requesting if {self.personality.name} needs to search internet to answer the user")
                    q = f"{self.separator_template}".join([
                        self.system_full_header,
                        f"Do you need internet search to answer the user prompt?"
                    ])
                    need = not self.personality.yes_no(q, self.user_custom_header("user") + current_message)
                    self.personality.step_end(f"Requesting if {self.personality.name} needs to search internet to answer the user")
                    self.personality.step("Yes" if need else "No")
                else:
                    need=True
                if need:
                    self.personality.step_start("Crafting internet search query")
                    q = f"{self.separator_template}".join([
                        f"{self.system_custom_header('discussion')}",
                        f"{discussion[-2048:]}",  # Use the last 2048 characters of the discussion for context
                        self.system_full_header,
                        f"You are a sophisticated web search query builder. Your task is to help the user by crafting a precise and concise web search query based on their request.",
                        f"Carefully read the discussion and generate a web search query that will retrieve the most relevant information to answer the last message from {self.config.user_name}.",
                        f"Do not answer the prompt directly. Do not provide explanations or additional information.",
                        f"{self.system_custom_header('current date')}{datetime.now()}",
                        f"{self.ai_custom_header('websearch query')}"
                    ])
                    query = self.personality.fast_gen(q, max_generation_size=256, show_progress=True, callback=self.personality.sink)
                    query = self.personality.remove_thinking_blocks(query)
                    query = query.replace("\"","")
                    self.personality.step_end("Crafting internet search query")
                    self.personality.step(f"web search query: {query}")

                    if self.config.internet_quick_search:
                        self.personality.step_start("Performing Internet search (quick mode)")
                    else:
                        self.personality.step_start("Performing Internet search (advanced mode: slower but more accurate)")

                    internet_search_results=f"{self.system_full_header}Use the web search results data to answer {self.config.user_name}. Try to extract information from the web search and use it to perform the requested task or answer the question. Do not come up with information that is not in the websearch results. Try to stick to the websearch results and clarify if your answer was based on the resuts or on your own culture. If you don't know how to perform the task, then tell the user politely that you need more data inputs.{self.separator_template}{self.start_header_id_template}Web search results{self.end_header_id_template}\n"

                    chunks:List[Chunk] = self.personality.internet_search_with_vectorization(query, self.config.internet_quick_search, asses_using_llm=self.config.activate_internet_pages_judgement)
                    
                    if len(chunks)>0:
                        for chunk in chunks:
                            internet_search_infos.append({
                                "title":chunk.doc.title,
                                "url":chunk.doc.path,
                                "brief":chunk.text
                            })
                            internet_search_results += self.system_custom_header("search result chunk")+f"\nchunk_infos:{chunk.doc.path}\nchunk_title:{chunk.doc.title}\ncontent:{chunk.text}\n"
                    else:
                        internet_search_results += "The search response was empty!\nFailed to recover useful information from the search engine.\n"
                    internet_search_results += self.system_custom_header("information") + "Use the search results to answer the user question."
                    if self.config.internet_quick_search:
                        self.personality.step_end("Performing Internet search (quick mode)")
                    else:
                        self.personality.step_end("Performing Internet search (advanced mode: slower but more advanced)")


        #User description
        user_description=""
        if self.config.use_user_informations_in_discussion:
            user_description=f"{self.start_header_id_template}User description{self.end_header_id_template}\n"+self.config.user_description+"\n"


        # Tokenize the conditionning text and calculate its number of tokens
        tokens_conditionning = self.model.tokenize(conditionning)
        n_cond_tk = len(tokens_conditionning)


        # Tokenize the internet search results text and calculate its number of tokens
        if len(internet_search_results)>0:
            tokens_internet_search_results = self.model.tokenize(internet_search_results)
            n_isearch_tk = len(tokens_internet_search_results)
        else:
            tokens_internet_search_results = []
            n_isearch_tk = 0


        # Tokenize the documentation text and calculate its number of tokens
        if len(documentation)>0:
            tokens_documentation = self.model.tokenize(documentation)
            n_doc_tk = len(tokens_documentation)
            self.info(f"The documentation consumes {n_doc_tk} tokens")
            if n_doc_tk>3*self.config.ctx_size/4:
                ASCIIColors.warning("The documentation is bigger than 3/4 of the context ")
                self.warning("The documentation is bigger than 3/4 of the context ")
            if n_doc_tk>=self.config.ctx_size-512:
                ASCIIColors.warning("The documentation is too big for the context")
                self.warning("The documentation is too big for the context it'll be cropped")
                documentation = self.model.detokenize(tokens_documentation[:(self.config.ctx_size-512)])
                n_doc_tk = self.config.ctx_size-512

        else:
            tokens_documentation = []
            n_doc_tk = 0



        # Tokenize user description
        if len(user_description)>0:
            tokens_user_description = self.model.tokenize(user_description)
            n_user_description_tk = len(tokens_user_description)
        else:
            tokens_user_description = []
            n_user_description_tk = 0


        function_calls = []
        if len(self.config.mounted_function_calls)>0:            
            for fc in self.config.mounted_function_calls:
                if fc["selected"]:
                    fci = self.load_function_call(fc, client)
                    if fci:
                        function_calls.append(fci)
        # Calculate the total number of tokens between conditionning, documentation, and knowledge
        total_tokens = n_cond_tk + n_isearch_tk + n_doc_tk + n_user_description_tk + n_positive_boost + n_negative_boost + n_fun_mode + n_think_first_mode

        # Calculate the available space for the messages
        available_space = self.config.ctx_size - n_tokens - total_tokens

        # if self.config.debug:
        #     self.info(f"Tokens summary:\nConditionning:{n_cond_tk}\nn_isearch_tk:{n_isearch_tk}\ndoc:{n_doc_tk}\nhistory:{n_history_tk}\nuser description:{n_user_description_tk}\nAvailable space:{available_space}",10)

        # Raise an error if the available space is 0 or less
        if available_space<1:
            ASCIIColors.red(f"available_space:{available_space}")
            ASCIIColors.red(f"n_doc_tk:{n_doc_tk}")
            
            ASCIIColors.red(f"n_isearch_tk:{n_isearch_tk}")
            
            ASCIIColors.red(f"n_tokens:{n_tokens}")
            ASCIIColors.red(f"self.config.max_n_predict:{self.config.max_n_predict}")
            self.InfoMessage(f"Not enough space in context!!\nVerify that your vectorization settings for documents or internet search are realistic compared to your context size.\nYou are {available_space} short of context!")
            raise Exception("Not enough space in context!!")

        # Accumulate messages until the cumulative number of tokens exceeds available_space
        tokens_accumulated = 0


        # Initialize a list to store the full messages
        full_message_list = []
        # If this is not a continue request, we add the AI prompt
        if not is_continue:
            message_tokenized = self.model.tokenize(
                self.personality.ai_message_prefix.strip()
            )
            full_message_list.append(message_tokenized)
            # Update the cumulative number of tokens
            tokens_accumulated += len(message_tokenized)


        if generation_type != "simple_question":
            # Accumulate messages starting from message_index
            for i in range(message_index, -1, -1):
                message = messages[i]

                # Check if the message content is not empty and visible to the AI
                if message.content != '' and (
                        message.message_type <= MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_SET_CONTENT_INVISIBLE_TO_USER.value and message.message_type != MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_SET_CONTENT_INVISIBLE_TO_AI.value):

                    if self.config.keep_thoughts:
                        content = message.content
                    else:
                        content = self.personality.remove_thinking_blocks(message.content)
                        
                    if message.sender_type == SENDER_TYPES.SENDER_TYPES_AI.value:
                        if self.config.use_assistant_name_in_discussion:
                            if self.config.use_model_name_in_discussions:
                                msg = self.ai_custom_header(message.sender+f"({message.model})") + content.strip()
                            else:
                                msg = self.ai_full_header + content.strip()
                        else:
                            if self.config.use_model_name_in_discussions:
                                msg = self.ai_custom_header("assistant"+f"({message.model})") + content.strip()
                            else:
                                msg = self.ai_custom_header("assistant") + content.strip()
                    else:
                        if self.config.use_user_name_in_discussions:
                            msg = self.user_full_header + content.strip()
                        else:
                            msg = self.user_custom_header("user") + content.strip()
                    msg += self.separator_template
                    message_tokenized = self.model.tokenize(msg)

                    # Check if adding the message will exceed the available space
                    if tokens_accumulated + len(message_tokenized) > available_space:
                        # Update the cumulative number of tokens
                        msg = message_tokenized[-(available_space-tokens_accumulated):]
                        tokens_accumulated += available_space-tokens_accumulated
                        full_message_list.insert(0, msg)
                        break

                    # Add the tokenized message to the full_message_list
                    full_message_list.insert(0, message_tokenized)

                    # Update the cumulative number of tokens
                    tokens_accumulated += len(message_tokenized)
        else:
            message = messages[message_index]

            # Check if the message content is not empty and visible to the AI
            if message.content != '' and (
                    message.message_type <= MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_SET_CONTENT_INVISIBLE_TO_USER.value and message.message_type != MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_SET_CONTENT_INVISIBLE_TO_AI.value):
                if self.config.keep_thoughts:
                    content = message.content
                else:
                    content = self.personality.remove_thinking_blocks(message.content)

                if message.sender_type == SENDER_TYPES.SENDER_TYPES_AI.value:
                    if self.config.use_assistant_name_in_discussion:
                        if self.config.use_model_name_in_discussions:
                            msg = self.ai_custom_header(message.sender+f"({message.model})") + content.strip()
                        else:
                            msg = self.ai_full_header + content.strip()
                    else:
                        if self.config.use_model_name_in_discussions:
                            msg = self.ai_custom_header("assistant"+f"({message.model})") + content.strip()
                        else:
                            msg = self.ai_custom_header("assistant") + content.strip()
                else:
                    if self.config.use_user_name_in_discussions:
                        msg = self.user_full_header + content.strip()
                    else:
                        msg = self.user_custom_header("user") + content.strip()
                message_tokenized = self.model.tokenize(msg)

                # Add the tokenized message to the full_message_list
                full_message_list.insert(0, message_tokenized)

                # Update the cumulative number of tokens
                tokens_accumulated += len(message_tokenized)

        # Build the final discussion messages by detokenizing the full_message_list
        discussion_messages = ""
        for i in range(len(full_message_list)-1 if not is_continue else len(full_message_list)):
            message_tokens = full_message_list[i]
            discussion_messages += self.model.detokenize(message_tokens)
        
        if len(full_message_list)>0:
            ai_prefix = self.personality.ai_message_prefix
        else:
            ai_prefix = ""


        # Details
        context_details = LollmsContextDetails(
            client=client,
            conditionning=conditionning,
            internet_search_infos=internet_search_infos,
            internet_search_results=internet_search_results,
            documentation=documentation,
            documentation_entries=documentation_entries,
            user_description=user_description,
            discussion_messages=discussion_messages,
            positive_boost=positive_boost,
            negative_boost=negative_boost,
            current_language=self.config.current_language,
            fun_mode=fun_mode,
            think_first_mode=think_first_mode,
            ai_prefix=ai_prefix,
            extra="",
            available_space=available_space,
            skills=skills_detials,
            is_continue=is_continue,
            previous_chunk=previous_chunk,
            prompt=current_message.content,
            function_calls=function_calls,

            debug= self.config.debug,
            ctx_size= self.config.ctx_size,
            max_n_predict= self.config.max_n_predict,

            model= self.model
        )
        
        
        if self.config.debug and not self.personality.processor:
            ASCIIColors.highlight(documentation,"source_document_title", ASCIIColors.color_yellow, ASCIIColors.color_red, False)
        # Return the prepared query, original message content, and tokenized query
        return context_details      


    # Properties ===============================================
    @property
    def start_header_id_template(self) -> str:
        """Get the start_header_id_template."""
        return self.config.start_header_id_template

    @property
    def end_header_id_template(self) -> str:
        """Get the end_header_id_template."""
        return self.config.end_header_id_template
    
    @property
    def system_message_template(self) -> str:
        """Get the system_message_template."""
        return self.config.system_message_template


    @property
    def separator_template(self) -> str:
        """Get the separator template."""
        return self.config.separator_template


    @property
    def start_user_header_id_template(self) -> str:
        """Get the start_user_header_id_template."""
        return self.config.start_user_header_id_template
    @property
    def end_user_header_id_template(self) -> str:
        """Get the end_user_header_id_template."""
        return self.config.end_user_header_id_template
    @property
    def end_user_message_id_template(self) -> str:
        """Get the end_user_message_id_template."""
        return self.config.end_user_message_id_template




    # Properties ===============================================
    @property
    def start_header_id_template(self) -> str:
        """Get the start_header_id_template."""
        return self.config.start_header_id_template

    @property
    def end_header_id_template(self) -> str:
        """Get the end_header_id_template."""
        return self.config.end_header_id_template
    
    @property
    def system_message_template(self) -> str:
        """Get the system_message_template."""
        return self.config.system_message_template


    @property
    def separator_template(self) -> str:
        """Get the separator template."""
        return self.config.separator_template


    @property
    def start_user_header_id_template(self) -> str:
        """Get the start_user_header_id_template."""
        return self.config.start_user_header_id_template
    @property
    def end_user_header_id_template(self) -> str:
        """Get the end_user_header_id_template."""
        return self.config.end_user_header_id_template
    @property
    def end_user_message_id_template(self) -> str:
        """Get the end_user_message_id_template."""
        return self.config.end_user_message_id_template




    @property
    def start_ai_header_id_template(self) -> str:
        """Get the start_ai_header_id_template."""
        return self.config.start_ai_header_id_template
    @property
    def end_ai_header_id_template(self) -> str:
        """Get the end_ai_header_id_template."""
        return self.config.end_ai_header_id_template
    @property
    def end_ai_message_id_template(self) -> str:
        """Get the end_ai_message_id_template."""
        return self.config.end_ai_message_id_template
    @property
    def system_full_header(self) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_header_id_template}{self.system_message_template}{self.end_header_id_template}"
    @property
    def user_full_header(self) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_user_header_id_template}{self.config.user_name}{self.end_user_header_id_template}"
    @property
    def ai_full_header(self) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_user_header_id_template}{self.personality.name}{self.end_user_header_id_template}"

    def system_custom_header(self, ai_name) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_user_header_id_template}{ai_name}{self.end_user_header_id_template}"

    def user_custom_header(self, ai_name) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_user_header_id_template}{ai_name}{self.end_user_header_id_template}"

    def ai_custom_header(self, ai_name) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_ai_header_id_template}{ai_name}{self.end_ai_header_id_template}"

