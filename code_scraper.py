import os
import json
from langchain.docstore.document import Document
from src.vector_store_SB import split_and_extract_words
from src.helper_func import llm, llm_mini, llm_mini_json
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser


def scrape_python_files_to_text(directory, output_file):
    """
    Recursively scrapes all Python files in a specified directory and its subdirectories,
    then writes their contents into a single text file.

    Args:
    directory (str): The root directory from which to start scraping Python files.
    output_file (str): The path to the text file where the content of all Python files will be written.
    """
    with open(output_file, "w") as outfile:
        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(directory):
            # Filter and process each Python file
            for filename in [
                f
                for f in filenames
                if f.endswith(".py") and not f.endswith("__init__.py")
            ]:
                filepath = os.path.join(dirpath, filename)
                filepath_cut ="src/" + filepath.split("pycram/src/")[-1]
                try:
                    with open(filepath, "r") as file:
                        content = file.read()
                        outfile.write(f"##New # Content from: #<{filepath_cut}>#\n")
                        outfile.write(content + "\n\n")
                except IOError as e:
                    print(f"Error reading file {filepath}: {e}")

def scrape_docu_files_to_text(directory, output_file):
    """
    Recursively scrapes all Python files in a specified directory and its subdirectories,
    then writes their contents into a single text file.

    Args:
    directory (str): The root directory from which to start scraping Python files.
    output_file (str): The path to the text file where the content of all Python files will be written.
    """
    with open(output_file, "w") as outfile:
        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(directory):
            # Filter and process each Python file
            for filename in [
                f
                for f in filenames
                if f.endswith(".md") and not f.endswith("README.md")
            ]:
                filepath = os.path.join(dirpath, filename)
                filename_cut = filename.split(".md")[0]
                try:
                    with open(filepath, "r") as file:
                        content = file.read()
                        outfile.write(f"##New # Content from: #<{filename_cut}>#\n")
                        outfile.write(content.split("---")[-1] + "\n\n")
                except IOError as e:
                    print(f"Error reading file {filepath}: {e}")

def scrape_udrf_files_to_text(directory, output_file):
    """
    Recursively scrapes all Python files in a specified directory and its subdirectories,
    then writes their contents into a single text file.

    Args:
    directory (str): The root directory from which to start scraping Python files.
    output_file (str): The path to the text file where the content of all Python files will be written.
    """
    with open(output_file, "w") as outfile:
        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(directory):
            # Filter and process each Python file
            for filename in [f for f in filenames if f.endswith(".urdf")]:
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, "r") as file:
                        content = file.read()
                        outfile.write(f"##New # Content from: #<{filename}>#\n")
                        outfile.write(content + "\n\n")
                except IOError as e:
                    print(f"Error reading file {filepath}: {e}")


# Code to Finetune data
def code_to_jsonl_finetuning(code, jsonl_path, prompt):
    with open(code, "r", encoding="utf-8") as file:
        content = file.read()
    split_parts = content.split("##New ")[1:]
    first_words = [part.split("\n")[0] if part.split() else "" for part in split_parts]
    parts = list(zip(first_words, split_parts))
    print(parts)
    print("\n----\n")
    finetune_prompt = ChatPromptTemplate.from_template(
        """Given the following context, generate a relevant user question that could be used to fine-tune a language model. The question should be directly related to the context provided and be realistic in terms of what a user might ask.

**Context:**
{context}

**Request:**
Please write a user question that aligns with the context above. The output should be a single question.
Example: {instruction}"""
    )

    # More complex template for tutorial writing, generating comprehensive documentation
    finetune_chain = (
        {
            "context": itemgetter("context"),
            "instruction": itemgetter("instruction"),
        }
        | finetune_prompt
        | llm_mini
        | StrOutputParser()
    )

    # Open the JSONL file for writing
    with open(jsonl_path, 'w') as jsonl_file:
        for part in parts:
            # Generate the fine-tuning result for each code part
            result = finetune_chain.invoke({"context": part[0], "instruction": prompt})
            finetuning_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a coding expert and know how to write PyCRAM Plan Code.",
                    },
                    {"role": "user", "content": result},
                    {
                        "role": "assistant",
                        "content": part[1],
                    },
                ]
            }

            # Write the example directly to the JSONL file as a new line
            json_line = json.dumps(finetuning_example, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')

# Code to Finetune data
def docu_to_jsonl_finetuning(docu, jsonl_path, prompt):
    first_words, split_parts = split_and_extract_words(docu)
    parts = list(zip(first_words, split_parts))
    print(parts)
    print("\n----\n")
    finetune_prompt = ChatPromptTemplate.from_template(
        """Given the following context, generate a relevant user question that could be used to fine-tune a language model. The question should be directly related to the context provided and be realistic in terms of what a user might ask.

**Context:**
{context}

**Request:**
Please write a user question that aligns with the context above. The output should be a single question.
Example: {instruction}"""
    )

    # More complex template for tutorial writing, generating comprehensive documentation
    finetune_chain = (
        {
            "context": itemgetter("context"),
            "instruction": itemgetter("instruction"),
        }
        | finetune_prompt
        | llm_mini
        | StrOutputParser()
    )

    # Open the JSONL file for writing
    with open(jsonl_path, 'w') as jsonl_file:
        for part in parts:
            # Generate the fine-tuning result for each code part
            context = part[0]
            if context.startswith("pycram."):
                context = "API Seite der PyCRAM Dokumentation:\n" + context
            else:
                context = "Tutorial Seite der PyCRAM Dokumentation:\n" + context
            result = finetune_chain.invoke({"context": {context}, "instruction": prompt})
            finetuning_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a coding expert and know how to write PyCRAM Plan Code.",
                    },
                    {"role": "user", "content": result},
                    {
                        "role": "assistant",
                        "content": part[1],
                    },
                ]
            }

            # Write the example directly to the JSONL file as a new line
            json_line = json.dumps(finetuning_example, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')

"""code_to_jsonl_finetuning(
    "output_code_new.txt",
    "finetune_code_new.jsonl",
    "What is written in the cache_manager.py file in the PyCRAM Framework code?",
)"""
"""
docu_to_jsonl_finetuning(
    "Documentation_important.txt",
    "finetune_docu.jsonl",
    "What is written in the Designator Tutorial Site in the PyCRAM Documentation?",
)"""
# scrape_python_files_to_text("/home/julius/ros/ros_ws/src/pycram/src/pycram", "output_code_new.txt")
# scrape_udrf_files_to_text("d:/Git/Repository/pycram/resources", "output_urdf_new.txt")
# scrape_docu_files_to_text("/home/julius/ros/ros_ws/src/pycram/examples", "output_doku.txt")
"""
result = chunk_json_for_llm("doc_code.json")
print(result)
print("-----")
print(len(result))
"""
