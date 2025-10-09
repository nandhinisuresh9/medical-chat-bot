from flask import Flask, request, jsonify, render_template
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
embeddings = download_embeddings()
docsearch=PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)
retriver=docsearch.as_retriever(search_type="similarity",search_kwargs={"k": 3})
chat_ai=ChatOpenAI(model_name="gpt-4o")

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
    print(response["answer"])
    return str(response["answer"])   
create_documents=create_stuff_documents_chain(chat_ai,chat_template)
create_retrieval_chain=create_retrieval_chain(retriver,create_documents)
@app.route("/")
def index():
    return render_template('chat.html')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)