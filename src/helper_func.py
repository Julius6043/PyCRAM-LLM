from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import re
import tiktoken
from vector_store_SB import get_retriever


llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
llm_o = ChatOpenAI(
    model="chatgpt-4o-latest",
    temperature=0,
)
llm_pycram = ChatOpenAI(
    model="ft:gpt-4o-2024-08-06:personal:pycram-v3:A6w6emoR",
    temperature=0,
)
llm_pycram_mini = ChatOpenAI(
    model="ft:gpt-4o-mini-2024-07-18:personal:pycram-v3-mini:A6w4KlUM",
    temperature=0,
)
llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_AH = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
llm_AO = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
llm_AS = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
llm_GP = ChatGoogleGenerativeAI(model="gemini-1.5-pro-exp-0827", temperature=0)
llm_json = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)
llm_mini_json = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)
llm_solver = llm_pycram
llm_planer = llm_GP
llm_tools = llm_pycram_mini
# llm_llama3 = ChatOllama(model="llama3")


# Function to clean and format documents, removing unwanted patterns and reducing whitespace
def format_docs(docs):
    text = "\n\n---\n\n".join([d.page_content for d in docs])
    pattern = r"Next \n\n.*?\nBuilds"
    pattern2 = r"pycram\n          \n\n                latest\n.*?Edit on GitHub"
    filtered_text = re.sub(pattern, "", text, flags=re.DOTALL)
    filtered_text2 = re.sub(pattern2, "", filtered_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", filtered_text2)
    return cleaned_text


def format_code(codes):
    text = "\n\n---\n\n".join([d.page_content for d in codes])
    return text


# Define a function to format documents for better readability
def format_examples(docs):
    # Join documents using a specified delimiter for separation
    return "\n\n<next example>\n".join([d.page_content for d in docs])


# Define a function to format documents for better readability
def format_example(example):
    text = example[0].page_content
    code_example = text.split("The corresponding plan")[0]
    return code_example


# Count the Tokens in a string
def count_tokens(model_name, text):
    # Lade das Tokenizer-Modell
    encoding = tiktoken.encoding_for_model(model_name)

    # Tokenisiere den Text
    tokens = encoding.encode(text)

    # Anzahl der Tokens
    return len(tokens)


# Function to get urdf content out of filenames in a string
def extract_urdf_files(input_string) -> str:
    """
    Extrahiert alle .urdf Dateinamen aus dem gegebenen String.
    Die Dateinamen können von einfachen oder doppelten Anführungszeichen umgeben sein
    oder einfach durch Leerzeichen getrennt sein.

    Args:
        input_string (str): Der zu durchsuchende String.

    Returns:
        list: Eine Liste der gefundenen .urdf Dateinamen.
    """
    # Regex-Muster, das drei Fälle abdeckt:
    # 1. Dateiname in einfachen Anführungszeichen
    # 2. Dateiname in doppelten Anführungszeichen
    # 3. Dateiname ohne Anführungszeichen, getrennt durch Leerzeichen oder Start/Ende des Strings
    pattern = r"""['"]([^'"]+\.urdf)['"]"""

    # Verwende re.VERBOSE für bessere Lesbarkeit und re.IGNORECASE für Groß-/Kleinschreibung
    matches = re.findall(pattern, input_string)

    file_content = ""
    for file in matches:
        urdf_retriever = get_retriever(4, 1, {"source": file})
        result = urdf_retriever.invoke(file)[0].page_content
        add_result = f"{result}\n\n-----\n"
        file_content += add_result
    return file_content
