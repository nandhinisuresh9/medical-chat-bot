from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
# extract text from PDF
def extract_text_from_pdf(data):
    Loader= DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents = Loader.load()
    return documents
def filter_to_min_docs(documents: List[Document]) -> List[Document]: 
    minimum_docs :List[Document] = []
    for doc in documents:
        src=doc.metadata.get("source")
        minimum_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    return minimum_docs

def text_split(minimaldocs: List[Document]) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    texts = text_splitter.split_documents(minimaldocs)
    return texts
def download_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings