import os

from dotenv import load_dotenv
from pydantic import SecretStr

from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompt_values import PromptValue



# from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

# Load the PDF document
loader = PyPDFLoader("../Tax-interview.pdf")
document = loader.load()

# Split the document into smaller chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(document)

# build the vectorstore using OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),  # Ensure your API key is set in the environment
)
vectorstore = FAISS.from_documents(docs, embeddings)


print(f"Number of documents in the vectorstore: {vectorstore.index.ntotal}\n\n")


# Create a retriever and LLM for question answering
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),  # Ensure your API key is set in the environment,
    temperature=0.0  # Set temperature to 0 for deterministic responses
)

# Create a RetrievalQA chain with the retriever and LLM
# this is legacy use
qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
query = "What is the main topic of the document?"
response = qa_chain.invoke(query)
print(response, "\n\n")


# This is the new way to use RetrievalQA with a retriever with LCEL

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = hub.pull("rlm/rag-prompt")

qa_chain = (
    {
        "context": vectorstore.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

response = qa_chain.invoke(query)
print(response, "\n\n")
