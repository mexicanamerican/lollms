from functools import partial
from typing import Dict, Any, List
from enum import Enum, auto
from lollms.client_session import Client

# Step 1: Define the FunctionType enum
class FunctionType(Enum):
    CONTEXT_UPDATE = auto()  # Adds information to the context
    AI_FIRST_CALL = auto()  # Called by the AI first, returns output, AI continues
    CLASSIC = auto()  # A classic function call with prompt

# Step 2: Update the FunctionCall base class
class FunctionCall:
    def __init__(self, function_type: FunctionType, client: Client, static_parameters=dict):
        self.function_type = function_type
        self.client = client
        self.static_parameters = static_parameters

    def execute(self, *args, **kwargs):
        """
        Execute the function based on its type.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")

    def update_context(self, context, contructed_context:str):
        """
        Update the context if needed.
        This method should be overridden by subclasses.
        """
        if self.function_type == FunctionType.CONTEXT_UPDATE:
            raise NotImplementedError("Subclasses must implement the update_context method for CONTEXT_UPDATE functions.")
        elif self.function_type == FunctionType.AI_FIRST_CALL:
            raise NotImplementedError("Subclasses must implement the update_context method for AI_FIRST_CALL functions.")
        elif self.function_type == FunctionType.POST_GENERATION:
            raise NotImplementedError("Subclasses must implement the update_context method for POST_GENERATION functions.")
        
    def process_output(self, context, llm_output:str):
        if self.function_type == FunctionType.CONTEXT_UPDATE:
            raise NotImplementedError("Subclasses must implement the process_output for CONTEXT_UPDATE functions.")



# from lollms.tasks import TasksLibrary
# class FunctionCalling_Library:
#     def __init__(self, tasks_library:TasksLibrary):
#         self.tl = tasks_library
#         self.function_definitions = []

#     def register_function(self, function_name, function_callable, function_description, function_parameters):
#         self.function_definitions.append({
#             "function_name": function_name,
#             "function": function_callable,
#             "function_description": function_description,
#             "function_parameters": function_parameters
#         })

#     def unregister_function(self, function_name):
#         self.function_definitions = [func for func in self.function_definitions if func["function_name"] != function_name]


#     def execute_function_calls(self, function_calls: List[Dict[str, Any]]) -> List[Any]:
#         """
#         Executes the function calls with the parameters extracted from the generated text,
#         using the original functions list to find the right function to execute.

#         Args:
#             function_calls (List[Dict[str, Any]]): A list of dictionaries representing the function calls.
#             function_definitions (List[Dict[str, Any]]): The original list of functions with their descriptions and callable objects.

#         Returns:
#             List[Any]: A list of results from executing the function calls.
#         """
#         results = []
#         # Convert function_definitions to a dict for easier lookup
#         functions_dict = {func['function_name']: func['function'] for func in self.function_definitions}

#         for call in function_calls:
#             function_name = call.get("function_name")
#             parameters = call.get("function_parameters", [])
#             function = functions_dict.get(function_name)

#             if function:
#                 try:
#                     # Assuming parameters is a dictionary that maps directly to the function's arguments.
#                     if type(parameters)==list:
#                         result = function(*parameters)
#                     elif type(parameters)==dict:
#                         result = function(**parameters)
#                     results.append(result)
#                 except TypeError as e:
#                     # Handle cases where the function call fails due to incorrect parameters, etc.
#                     results.append(f"Error calling {function_name}: {e}")
#             else:
#                 results.append(f"Function {function_name} not found.")

#         return results
    
#     def generate_with_functions(self, prompt):
#         # Assuming generate_with_function_calls is a method from TasksLibrary
#         ai_response, function_calls = self.tl.generate_with_function_calls(prompt, self.function_definitions)
#         return ai_response, function_calls

#     def generate_with_functions_with_images(self, prompt, image_files):
#         # Assuming generate_with_function_calls_and_images is a method from TasksLibrary
#         if len(image_files) > 0:
#             ai_response, function_calls = self.tl.generate_with_function_calls_and_images(prompt, image_files, self.function_definitions)
#         else:
#             ai_response, function_calls = self.tl.generate_with_function_calls(prompt, self.function_definitions)

#         return ai_response, function_calls