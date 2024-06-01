# Lollms function call definition file
# Here you need to import any necessary imports depending on the function requested by the user
import markdown
from pathlib import Path

# Partial is useful if we need to preset some parameters
from functools import partial

# It is advised to import typing elements
from typing import List

# Import PackageManager if there are potential libraries that need to be installed 
from lollms.utilities import PackageManager

# ascii_colors offers advanced console coloring and bug tracing
from ascii_colors import trace_exception

# Here is an example of how we install a non installed library using PackageManager
if not PackageManager.check_package_installed("markdown2latex"):
    PackageManager.install_package("markdown2latex")

# now we can import the library
import markdown2latex

# Function to convert markdown file to LaTeX file
def markdown_file_to_latex(file_path: str) -> str:
    try:
        # handle exceptions

        # Load the markdown file
        markdown_text = Path(file_path).read_text()

        # Convert markdown to latex
        latex_text = markdown2latex.convert(markdown_text)
        
        # Define output file path
        output_path = Path(file_path).with_suffix('.tex')
        
        # Save the latex text to a file
        output_path.write_text(latex_text)
        
        # Finally we return the path to the LaTeX file
        return str(output_path)
    except Exception as e:
        return trace_exception(e)

# Function to convert markdown string to LaTeX string
def markdown_string_to_latex(markdown_text: str) -> str:
    try:
        # handle exceptions

        # Convert markdown to latex
        latex_text = markdown2latex.convert(markdown_text)
        
        # Finally we return the LaTeX text
        return latex_text
    except Exception as e:
        return trace_exception(e)

# Metadata function for markdown_file_to_latex
def markdown_file_to_latex_function():
    return {
        "function_name": "markdown_file_to_latex", # The function name in string
        "function": markdown_file_to_latex, # The function to be called
        "function_description": "Converts a markdown file to a LaTeX file.", # Description
        "function_parameters": [{"name": "file_path", "type": "str"}] # The set of parameters
    }

# Metadata function for markdown_string_to_latex
def markdown_string_to_latex_function():
    return {
        "function_name": "markdown_string_to_latex", # The function name in string
        "function": markdown_string_to_latex, # The function to be called
        "function_description": "Converts a markdown string to a LaTeX string.", # Description
        "function_parameters": [{"name": "markdown_text", "type": "str"}] # The set of parameters
    }
