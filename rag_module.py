# RAG Module for Document Q&A
import os
from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

class RAG:
    def __init__(self, model_name: str = "gemini-2.0-flash-001", embedding_model: str = "models/embedding-001"):
        """Initialize the RAG system with model and embedding model."""
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.vectorstore = None
        self.qa_chain = None
        self.curr_doc = None
        self.is_loaded = False

    def load_document(self, doc_path: str):
        try:
            if not os.path.exists(doc_path):
                return {
                    "success": False,
                    "message": f"File not found: {doc_path}"
                }
            
            if ".csv" in doc_path:
                loader = CSVLoader(file_path=doc_path)
            elif ".pdf" in doc_path:
                loader = PyPDFLoader(file_path=doc_path)
            elif ".txt" in doc_path:
                loader = TextLoader(file_path=doc_path)
            else:
                return {
                    "success": False,
                    "message": "Unsupported file type. Please provide a CSV, PDF, or TXT file."
                }

            docs = loader.load()
            
            if not docs:
                return {
                    "success": False,
                    "message": "The document appears to be empty or couldn't be loaded."
                }
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_documents(docs)

            self.vectorstore = Chroma.from_documents(chunks, self.embeddings)

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=False
            )

            self.curr_doc = doc_path
            self.is_loaded = True

            return {
                "success": True,
                "message": f"Document '{os.path.basename(doc_path)}' loaded successfully!",
                "document": doc_path
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error loading document: {str(e)}"
            }

    def ask_question(self, question: str):
        if not self.is_loaded or not self.qa_chain:
            return {
                "success": False,
                "message": "No document loaded. Please load a document first."
            }
        
        try:
            response = self.qa_chain.invoke({"query": question})
            
            if isinstance(response, dict) and "result" in response:
                answer = response["result"]
            else:
                answer = str(response)
            
            return {
                "success": True,
                "answer": answer
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing question: {str(e)}"
            }

    def get_status(self):
        return {
            "is_loaded": self.is_loaded,
            "current_document": os.path.basename(self.curr_doc) if self.curr_doc else None
        }

    def clear_document(self):
        self.vectorstore = None
        self.qa_chain = None
        self.curr_doc = None
        self.is_loaded = False
        return {"success": True, "message": "Document cleared."}

# Global instance
_rag_instance: Optional[RAG] = None

def get_rag_instance() -> RAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAG()
    return _rag_instance

def load_document_for_qa(doc_path: str) -> Dict[str, Any]:
    rag = get_rag_instance()
    return rag.load_document(doc_path)

def ask_document_question(question: str) -> Dict[str, Any]:
    rag = get_rag_instance()
    return rag.ask_question(question)

def get_document_status() -> Dict[str, Any]:
    rag = get_rag_instance()
    return rag.get_status()

def clear_current_document() -> Dict[str, Any]:
    rag = get_rag_instance()
    return rag.clear_document()
