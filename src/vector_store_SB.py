# Load environment variables from a .env file for secure access to sensitive data like API keys.
from dotenv import load_dotenv

# Import libraries and modules for document loading, text splitting, and vector storage.
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from supabase.client import Client, create_client
from supabase.lib.client_options import ClientOptions
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup as Soup
import os
import httpx

# Initialize environment variables.
load_dotenv()

# Configuration for the Supabase client.
client_options = ClientOptions(postgrest_client_timeout=None)
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

# Create a Supabase client using the URL and key from environment variables.
supabase: Client = create_client(supabase_url, supabase_key, options=client_options)

# Initialize OpenAI embeddings for vectorization of texts.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings_large = OpenAIEmbeddings(model="text-embedding-3-large")

# Create vector stores for storing and querying vectorized text using Supabase and the specified embeddings.
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

vector_store_large = SupabaseVectorStore(
    embedding=embeddings_large,
    client=supabase,
    table_name="docs",
    query_name="match_docs",
)

vector_store_examples = SupabaseVectorStore(
    embedding=embeddings_large,
    client=supabase,
    table_name="examples",
    query_name="match_examples",
)


# Function to extract text from a PDF file.
def get_pdf_text(file_name, path_pdf=True):
    text = ""
    if path_pdf:
        path = f"pdf\{file_name}"
    else:
        path = file_name
    reader = PdfReader(path)
    for page in reader.pages:
        text += page.extract_text()

    # Split the extracted text into manageable chunks for processing.
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to recursively load text content from a website up to a specified depth.
def load_website(link, max_depth=20):
    url = link
    loader = RecursiveUrlLoader(
        url=url, max_depth=max_depth, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs_pycram = loader.load()
    d_sorted = sorted(docs_pycram, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))
    return d_reversed


# Function to load and split text from a PDF file using PyMuPDFLoader.
def load_pdf(file_name):
    path = f"pdf\{file_name}"
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


# Similar to load_pdf but for a specific document path.
def load_pdf_document(file_name):
    path = file_name
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=1250, chunk_overlap=150, length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


# Function to load text into one of the vector stores.
def load_in_vector_store(source, vectore_store_id=1):
    global vector_store, vector_store_large, vector_store_examples
    if vectore_store_id == 1:
        chunks = load_pdf_document(source)
        vector_store = SupabaseVectorStore.from_documents(
            chunks,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=500,
        )
    elif vectore_store_id == 2:
        chunks = load_website(source)
        vector_store_large = SupabaseVectorStore.from_documents(
            chunks,
            embeddings_large,
            client=supabase,
            table_name="docs",
            query_name="match_docs",
        )
    elif vectore_store_id == 3:
        chunks = source
        vector_store_examples = SupabaseVectorStore.from_texts(
            chunks,
            embeddings_large,
            client=supabase,
            table_name="examples",
            query_name="match_examples",
        )
    else:
        raise Exception("Invalid vector store id")


# Function to delete entries from a vector store.
def delete_from_vectorstore(table, num=2):
    if num == -1:
        result = supabase.table("documents").select("id", count="exact").execute()
        num = result.data
    data = (
        supabase.table("documents")
        .select("id")
        .order("id", desc=True)
        .limit(num)
        .execute()
    )
    if data.data:
        ids_to_delete = [item["id"] for item in data.data]  # Extract IDs to delete.
        delete_response = (
            supabase.table(table).delete().in_("id", ids_to_delete).execute()
        )
    else:
        print("No entries found to delete.")


# Function to retrieve documents similar to a query from a vector store.
def retrieve(query):
    matched_docs = vector_store.similarity_search(query)
    result = matched_docs[0].page_content + "\n\n" + matched_docs[1].page_content
    return result


# Function to retrieve a specified number of documents similar to a query from the large vector store.
def retrieve_large(query, number):
    matched_docs = vector_store_large.similarity_search(query)
    result = ""
    for i in range(number):
        result += (
            "Document: " + str(i + 1) + "\n" + matched_docs[i].page_content + "\n\n"
        )
    return result


# Function to get a retriever based on the vector store ID and number of documents to retrieve.
def get_retriever(vector_store_id=1, num=5):
    vector_store_temp = vector_store
    if vector_store_id == 1:
        vector_store_temp = vector_store
    elif vector_store_id == 2:
        vector_store_temp = vector_store_large
    elif vector_store_id == 3:
        vector_store_temp = vector_store_examples
    else:
        raise Exception("Invalid vector store id")
    retriever = vector_store_temp.as_retriever(search_kwargs={"k": num})
    return retriever


# delete_from_vectorstore(table="documents", num=-1)
# load_in_vector_store("BAfoeG.pdf", 1)
# chunks = get_pdf_text("BAfoeG.pdf")
# load_in_vector_store("BAfoeG.pdf", 2)
# result = retrieve_large("Was muss ich erfüllen um Bafoeg beantragen zu können?")
# print(result)
# load_in_vector_store("https://pycram.readthedocs.io/en/latest/", 2)
