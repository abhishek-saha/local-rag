# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def main():
    # 1. Load the local document (Replace with your PDF path)
    pdf_path = "Franz Kafka - The Metamorphosis.pdf"
    if not os.path.exists(pdf_path):
        print(f"Please place a PDF file at '{pdf_path}' or update the path in the script.")
        return

    print("📄 Loading document...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split the document into chunks
    print("✂️ Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 3. Create Vector Store and Embeddings locally
    print("🧠 Generating embeddings and saving to Vector DB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 chunks

    # 4. Initialize Local LLM
    llm = Ollama(model="llama3", temperature=0)

    # 5. Define the RAG Prompt
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Context:
    {context}

    Question: 
    {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 6. Construct the RAG Chain
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # 7. Query the system
    print("\n🚀 RAG System Ready!")
    while True:
        query = input("\nAsk a question about your document (or type 'exit'): ")
        if query.lower() == 'exit':
            break

        print("\nThinking...")
        response = rag_chain.invoke(query)
        print(f"\nAnswer:\n{response}")


if __name__ == "__main__":
    main()
