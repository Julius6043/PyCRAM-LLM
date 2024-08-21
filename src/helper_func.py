from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import re
import tiktoken


llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
llm_o = ChatOpenAI(model="chatgpt-4o-latest", temperature=0,)
llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_AH = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
llm_AO = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
llm_AS = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
llm_GP = ChatGoogleGenerativeAI(model="gemini-1.5-pro-exp-0801", temperature=0)
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