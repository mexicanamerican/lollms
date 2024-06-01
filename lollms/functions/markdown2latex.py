# Import necessary libraries
import re
from pathlib import Path
from functools import partial
from typing import List
from ascii_colors import trace_exception

# Define the conversion function
def markdown_to_latex(file_path: str) -> str:
    try:
        # Load the markdown file
        markdown_text = Path(file_path).read_text()
        
        # Define conversion rules from markdown to LaTeX
        conversion_rules = [
            (r'\\', r'\\textbackslash{}'),  # Escape backslashes
            (r'([#]+) (.*)', lambda m: '\\' + 'sub'*(len(m.group(1))-1) + 'section{' + m.group(2) + '}'),  # Convert headings
            (r'\*\*(.*?)\*\*', r'\\textbf{\1}'),  # Bold text
            (r'\*(.*?)\*', r'\\textit{\1}'),  # Italic text
            (r'\!\[(.*?)\]\((.*?)\)', r'\\begin{figure}[h!]\n\\centering\n\\includegraphics[width=\\textwidth]{\2}\n\\caption{\1}\n\\end{figure}'),  # Images
            (r'\[(.*?)\]\((.*?)\)', r'\\href{\2}{\1}'),  # Links
            (r'`([^`]*)`', r'\\texttt{\1}'),  # Inline code
            (r'^```\s*([a-z]*)\s*\n([\s\S]*?)\n```', r'\\begin{verbatim}\2\\end{verbatim}'),  # Code blocks
            (r'^-\s+(.*)', r'\\begin{itemize}\n\\item \1\n\\end{itemize}'),  # Unordered lists
            (r'^\d+\.\s+(.*)', r'\\begin{enumerate}\n\\item \1\n\\end{enumerate}'),  # Ordered lists
            (r'^>(.*)', r'\\begin{quote}\1\\end{quote}'),  # Block quotes
        ]
        
        # Apply conversion rules
        latex_text = markdown_text
        for pattern, replacement in conversion_rules:
            latex_text = re.sub(pattern, replacement, latex_text, flags=re.MULTILINE)

        # Define output file path
        output_path = Path(file_path).with_suffix('.tex')
        
        # Save the LaTeX text to a file
        output_path.write_text(latex_text)
        
        # Finally we return the path to the LaTeX file
        return str(output_path)
    except Exception as e:
        return trace_exception(e)

# Metadata function
def markdown_to_latex_function():
    return {
        "function_name": "markdown_to_latex",  # The function name in string
        "function": markdown_to_latex,  # The function to be called
        "function_description": "Converts a markdown file to a LaTeX file.",  # Description
        "function_parameters": [{"name": "file_path", "type": "str"}]  # The set of parameters
    }
