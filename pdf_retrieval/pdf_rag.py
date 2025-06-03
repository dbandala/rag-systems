import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA



# Ensure you have the necessary libraries installed:
loader = PyPDFLoader("../Tax-interview.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)



embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")  # Ensure you have set your OpenAI API key in the environment variables
)
vectorstore = FAISS.from_documents(docs, embeddings)




retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY")  # Ensure your API key is set in the environment
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


query = "What is the main topic of the document?"
response = qa_chain.run(query)
print(response)

