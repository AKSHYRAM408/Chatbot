import streamlit as st
import os
import warnings
from typing import List
from dotenv import load_dotenv
import pickle
import io
import requests
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from bs4 import BeautifulSoup

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

# Function to load and process PDF
def load_pdf(file_path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        st.error(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

# Function to load and process URL
def load_url(url: str) -> List[Document]:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text()

        if not content.strip():
            st.error(f"Error: The content from the URL {url} is empty.")
            return []

        document = Document(page_content=content, metadata={"source": url})
        return [document]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL content: {e}")
        return []
    except Exception as e:
        st.error(f"Error loading content from URL {url}: {e}")
        return []

# Function to load and process Excel or CSV files
def load_excel(file_path: str) -> List[Document]:
    try:
        # Use pandas to read Excel or CSV
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            st.error(f"Unsupported file format: {file_path}")
            return []

        content = df.to_string(index=False)  # Convert the dataframe to a string
        document = Document(page_content=content, metadata={"source": file_path})
        return [document]
    except Exception as e:
        st.error(f"Error processing Excel/CSV: {e}")
        return []

# Function to split documents into chunks
def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# Function to create a vector store using FAISS
def create_vector_store(documents: List[Document], embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vector_store = FAISS.from_documents(documents, embeddings)

        with open("faiss_vector_store.pkl", "wb") as f:
            pickle.dump(vector_store, f)

        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Function to load the existing vector store from a file
def load_vector_store():
    try:
        with open("faiss_vector_store.pkl", "rb") as f:
            vector_store = pickle.load(f)
        return vector_store
    except FileNotFoundError:
        return None

# Function to create the question-answering chain
def create_qa_chain(vector_store):
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_retries=2
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Return top 5 similar documents
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

class DocumentProcessor:
    def __init__(self):
        self.vector_store = load_vector_store()
        self.qa_chain = None
        self.processed_files = []
    
    # Process PDFs, URLs, and Excel files
    def process_documents(self, file_paths: List[str], url: str = ""):
        all_documents = []
        if url:
            documents = load_url(url)
            if documents:
                all_documents.extend(documents)
        
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                documents = load_pdf(file_path)
            elif file_path.endswith((".csv", ".xlsx")):
                documents = load_excel(file_path)
            else:
                st.error(f"Unsupported file type: {file_path}")
                continue

            if not documents:
                st.error(f"Failed to load file: {file_path}")
                continue

            all_documents.extend(documents)

        if all_documents:
            #st.info(f"Splitting {len(all_documents)} documents...")
            split_docs = split_documents(all_documents)

            if self.vector_store is None:
                self.vector_store = create_vector_store(split_docs)
            else:
                embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                self.vector_store.add_documents(split_docs)

            if not self.vector_store:
                st.error("Failed to create vector store")
                return "Failed to create vector store"
            
            #st.info("Creating QA chain...")
            self.qa_chain = create_qa_chain(self.vector_store)
            if not self.qa_chain:
                st.error("Failed to create QA chain")
                return "Failed to create QA chain"
            
            self.processed_files.extend(file_paths)
            #st.success(f"Successfully processed {len(file_paths)} file(s). Total processed files: {len(self.processed_files)}")
        else:
            st.warning("No valid documents found for processing.")
    
    # Query the processed documents
    def query_documents(self, query: str):
        if not self.qa_chain:
            return "Please process documents first.", []
        
        try:
            response = self.qa_chain.invoke({"query": query})
            if not response:
                return "No response from the QA chain.", []
            
            return response['result'], []
        
        except Exception as e:
            st.error(f"Error processing query: {e}")
            return f"Error processing query: {e}", []

def main():
    st.title("Akshy's AI Assisstant")
    st.markdown("### Search for Details of students.")

    if "document_processor" not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    document_processor = st.session_state.document_processor

    # Define the file paths you want to process
    file_paths = [
        ".\Documents\\ak1.xlsx",  # Path to your Excel file
        ".\Documents\\Zoho Resume Manjineshwaran.pdf",  # Path to your PDF file
    ]
    
    # URL to be processed
    url = "https://brainlox.com/courses/category/technical"

    document_processor.process_documents(file_paths, url)
    
    # Query Section
    query = st.text_input("Ask a Question", "")
    if st.button("Submit Query"):
        if query.strip():
            answer, sources = document_processor.query_documents(query)
            if answer:
                st.text_area("Answer", value=answer, height=100, disabled=True)
            else:
                st.warning("No answer returned. Ensure files are processed correctly.")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
