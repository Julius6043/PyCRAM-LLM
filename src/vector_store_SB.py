from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from supabase.client import Client, create_client
from supabase.lib.client_options import ClientOptions
from PyPDF2 import PdfReader
import httpx

load_dotenv()

client_options = ClientOptions(postgrest_client_timeout=None)
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key, options=client_options)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings_large = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

vector_store_large = SupabaseVectorStore(
    embedding=embeddings_large,
    client=supabase,
    table_name="data",
    query_name="match_data",
)


def get_pdf_text(file_name, path_pdf=True):
    text = ""
    if path_pdf:
        path = f"pdf\{file_name}"
    else:
        path = file_name
    reader = PdfReader(path)
    for page in reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def load_pdf(file_name):
    path = f"pdf\{file_name}"
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def load_pdf_document(file_name):
    path = file_name
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=1250, chunk_overlap=150, length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


# Laden in ein der beiden VecktorStores
def load_in_vector_store(file_name, vectore_store_id=1):
    global vector_store, vector_store_large
    if vectore_store_id == 1:
        chunks = load_pdf_document(file_name)
    else:
        chunks = load_pdf(file_name)
    # pages = load_pdf(file_name)
    if vectore_store_id == 1:
        vector_store = SupabaseVectorStore.from_documents(
            chunks,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=500,
        )
    elif vectore_store_id == 2:
        vector_store_large = SupabaseVectorStore.from_documents(
            chunks,
            embeddings_large,
            client=supabase,
            table_name="data",
            query_name="match_data",
            chunk_size=500,
        )
    else:
        raise Exception("Invalid vector store id")


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
        ids_to_delete = [item["id"] for item in data.data]  # IDs extrahieren
        delete_response = (
            supabase.table(table).delete().in_("id", ids_to_delete).execute()
        )
    else:
        print("Keine Einträge zum Löschen gefunden.")


def retrieve(query):
    matched_docs = vector_store.similarity_search(query)
    result = matched_docs[0].page_content + "\n\n" + matched_docs[1].page_content
    return result


def retrieve_large(query):
    matched_docs = vector_store_large.similarity_search(query)
    result = (
        matched_docs[0].page_content
        + "\n\n"
        + "Dokument: 2\n"
        + matched_docs[1].page_content
        + "\n\n"
        + "Dokument: 3\n"
        + matched_docs[2].page_content
        + "\n\n"
        + "Dokument: 4\n"
        + matched_docs[3].page_content
    )
    return result


def get_retriever(vector_store_id=1):
    vector_store_temp = vector_store
    if vector_store_id == 1:
        vector_store_temp = vector_store
    elif vector_store_id == 2:
        vector_store_temp = vector_store_large
    else:
        raise Exception("Invalid vector store id")
    retriever = vector_store_temp.as_retriever(search_kwargs={"k": 5})
    return retriever


# delete_from_vectorstore(table="documents", num=-1)
# load_in_vector_store("BAfoeG.pdf", 1)
# chunks = get_pdf_text("BAfoeG.pdf")
# load_in_vector_store("BAfoeG.pdf", 2)
# result = retrieve_large("Was muss ich erfüllen um Bafoeg beantragen zu können?")
# print(result)
