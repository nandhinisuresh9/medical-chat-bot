from flask import Flask, request, jsonify, render_template
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.prompt import *
from langchain_community.vectorstores import Milvus
from pymilvus import MilvusClient
from pymilvus import connections,utility
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from operator import itemgetter
HOST = "127.0.0.1"  # Or "localhost"
PORT = "19530"

# Connect to the Milvus server
connections.connect(
    alias="default", 
    host=HOST, 
    port=PORT
)

print(f"Connection established to Milvus at {HOST}:{PORT}")
client = MilvusClient(
    uri="http://localhost:19530",
    # token="root:Milvus" # Use this if your Milvus instance requires authentication
)
MILVUS_URI = "http://localhost:19530"
collection_name ="my_vector_collection505"

print("client created")
# Get a list of all collection names
all_collections = utility.list_collections()

print("Existing Collections:")
for col_name in all_collections:
    print(f"- {col_name}")
from pymilvus import Collection
collection = Collection("my_vector_collection505")
collection.load()
actual_text_field_name = None
        
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
import json # You might need this if your metadata is stored as a JSON string
embeddings = download_embeddings()

import os
from groq import Groq
app = Flask(__name__)
load_dotenv()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ['GROQ_API_KEY'] = GROQ_API_KEY

groqapi = ChatGroq(
            groq_api_key= GROQ_API_KEY, 
            model_name="llama-3.3-70b-versatile"
        )


vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={
            "uri": MILVUS_URI,
            #"token": MILVUS_TOKEN, # Uncomment if using a token (e.g., Zilliz Cloud)
        },
    primary_field="id",
    vector_field="values",
    metadata_field="metadata",
    text_field="text"
    
       )

retriver = vectorstore.as_retriever( 
    search_kwargs={
        "k": 3 # Example: set number of results, but NO 'output_fields'
    })
chat_template= ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

@app.route("/get", methods=["GET","POST"]  )
def chat():
    msg=request.form["msg"]
    print(msg)
    response=create_retrieval_chain.invoke({"input":msg})
    print(str(response["answer"]))
    return str(response["answer"])       
create_documents=create_stuff_documents_chain(groqapi,chat_template)
create_retrieval_chain=create_retrieval_chain(retriver,create_documents)
@app.route("/")
def index():
    return render_template('chat.html')
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)