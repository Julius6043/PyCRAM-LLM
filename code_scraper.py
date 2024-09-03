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
                try:
                    with open(filepath, "r") as file:
                        content = file.read()
                        outfile.write(f"##New # Content from: #<{filename}>#\n")
                        outfile.write(content + "\n\n")
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


### Json in Vektorstore
def chunk_json_for_llm(json_path):
    """
    This function takes a JSON file describing a framework and chunks it into smaller, meaningful pieces for LLM consumption,
    specifically tailored to the structure of the provided JSON, without using a chunk size limit.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list: A list of strings, each representing a meaningful chunk of the JSON data.
    """

    with open(json_path, "r") as f:
        framework_data = json.load(f)

    chunks = []

    # Chunk frameworkName, version, and description
    initial_chunk = f"Framework: {framework_data['frameworkName']}\nVersion: {framework_data['version']}\nDescription: {framework_data['description']}"
    chunks.append(initial_chunk)

    # Chunk keyConcepts
    chunks.extend(chunk_list_section(framework_data["keyConcepts"], "Key Concepts"))

    # Chunk components
    chunks.extend(chunk_list_section(framework_data["components"], "Components"))

    # Chunk api
    for api_category in framework_data["api"]:
        chunks.append(chunk_api_category(api_category))

    # Chunk examples
    chunks.extend(chunk_list_section(framework_data["examples"], "Examples"))

    # Chunk bestPractices
    chunks.extend(chunk_list_section(framework_data["bestPractices"], "Best Practices"))

    return chunks


def chunk_list_section(data_list, section_name):
    """
    Chunks a list section (like keyConcepts, components, examples, bestPractices) into individual items.

    Args:
        data_list (list): The list of data to be chunked.
        section_name (str): The name of the section.

    Returns:
        list: A list of strings, each representing an item from the data_list.
    """

    chunks = []
    for item in data_list:
        item_str = f"## {section_name}: {item['name']}\n{item['description']}\n"
        if "props" in item:
            item_str += f"Properties: {', '.join(item['props'])}\n"
        if "example" in item:
            item_str += f"Example: {item['example']}\n"
        chunks.append(item_str)

    return chunks


def chunk_api_category(api_category):
    """
    Creates a single chunk for an API category, including all its methods.

    Args:
        api_category (dict): The API category dictionary.

    Returns:
        str: A string representing the entire API category.
    """

    chunk = f"## API: {api_category['category']}\n"
    for method in api_category["methods"]:
        chunk += f"- {method['name']}: {method['description']}\n"
        if "parameters" in method:
            chunk += "  - Parameters:\n"
            for param in method["parameters"]:
                chunk += (
                    f"    - {param['name']} ({param['type']}): {param['description']}\n"
                )
        if "returns" in method:
            chunk += f"  - Returns: {method['returns']}\n"
        if "returnDescription" in method:
            chunk += f"    - {method['returnDescription']}\n"

    return chunk


# Code to Finetune data
def code_to_jsonl_finetuning(code, jsonl_path, prompt):
    first_words, split_parts = split_and_extract_words(code)
    parts = list(zip(first_words, split_parts))
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
    list_for_jsonl = []
    for part in parts:
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
        list_for_jsonl.append(finetuning_example)
    # Save the list of fine-tuning examples to a JSONL file
    with open(jsonl_path, "w") as jsonl_file:
        for example in list_for_jsonl:
            jsonl_file.write(json.dumps(example) + "\n")


code_to_jsonl_finetuning(
    "output_code_new.txt",
    "finetune_code_new.jsonl",
    "What is written in the cache_manager.py file in the PyCRAM Framework code?",
)
# scrape_python_files_to_text("d:/Git/Repository/pycram/src/pycram", "output_code_new.txt")
# scrape_udrf_files_to_text("d:/Git/Repository/pycram/resources", "output_urdf_new.txt")
"""
result = chunk_json_for_llm("doc_code.json")
print(result)
print("-----")
print(len(result))
"""
