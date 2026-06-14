# LangChain Documentation

## What is LangChain
LangChain is a framework for developing applications powered by large language models. It provides tools for chaining LLM calls, memory, and retrieval.

## Installation
pip install langchain langchain-community

## Basic LLM Call
Make a simple call to an LLM:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key="your-api-key",
    model="llama3-8b-8192",
)

response = llm.invoke("What is Python?")
print(response.content)
```

## Prompt Templates
Create reusable prompt templates:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}"),
])

chain = prompt | llm
response = chain.invoke({"question": "What is RAG?"})
```

## Document Loaders
Load documents from various sources:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("my_document.txt")
documents = loader.load()
```

## Text Splitters
Split documents into chunks:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)
```

## Vector Stores
Store and retrieve document embeddings:

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
```

## Retrieval Chain
Build a retrieval chain for Q&A:

```python
from langchain.chains import RetrievalQA

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)

answer = qa_chain.invoke("How do I use FastAPI?")
```

## Memory
Add conversation memory to your chain:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)
```

## Output Parsers
Parse LLM output into structured formats:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
chain = prompt | llm | parser
result = chain.invoke({"question": "List 3 Python frameworks"})
```