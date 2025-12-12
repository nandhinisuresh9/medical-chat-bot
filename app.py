from flask import Flask, request, jsonify, render_template
from src.helper import download_embeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.prompt import *
from langchain_community.vectorstores import Milvus
from pymilvus import MilvusClient
from pymilvus import connections,utility
import mlflow


HOST = "127.0.0.1"  # Or "localhost"
PORT = "19530"
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
mlflow.groq.autolog() 
mlflow.set_tracking_uri("http://localhost:5000") 
mlflow.set_experiment("Groq_RAG_Milvus_App2")

print(f"MLflow Autologging enabled for Groq. Tracking to: {mlflow.get_tracking_uri()}")
MILVUS_URI = "http://localhost:19530"
collection_name ="my_vector_collection80"
from langchain_community.vectorstores import Milvus
import os
from groq import Groq
app = Flask(__name__)
load_dotenv()
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
mlflow.groq.autolog(disable=True)
embeddings = download_embeddings()
print("Generating embeddings without MLflow autologging...")
mlflow.groq.autolog() 
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
create_documents=create_stuff_documents_chain(groqapi,chat_template)
create_retrieval_chain=create_retrieval_chain(retriver,create_documents)
@app.route("/get", methods=["GET","POST"]  )
def chat():
    msg=request.form["msg"]
   
    response=create_retrieval_chain.invoke({"input":msg})
    return str(response)       

@app.route("/")
def index():
    return render_template('chat.html')
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)