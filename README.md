# Customer-Support-Agent

A multi-modal customer support agent that combines LLM-powered chat, document Q&A (RAG), order tracking, complaint management, and escalation, with both API and Streamlit web interfaces.

## Features

- **Conversational AI**: Chat with an LLM-powered assistant for FAQs, order tracking, and more.
- **Order Tracking**: Check order status by order ID.
- **Complaint Management**: File complaints about orders and escalate unresolved issues.
- **Document Q&A (RAG)**: Load CSV, PDF, or TXT documents and ask questions about their content.
- **Streamlit Web App**: User-friendly interface for all features.
- **REST API**: FastAPI endpoints for complaints, order status, and escalations.

## File Overview

- `streamlit_customer_service.py`: Streamlit app for chat, order tracking, complaints, document Q&A, and escalation.
- `customer_agent.py`: Core logic for intent extraction, FAQ, complaint, order tracking, escalation, and RAG integration.
- `rag_module.py`: RAG (Retrieval-Augmented Generation) module for document loading, Q&A, and status.
- `api.py`: FastAPI backend for complaints, order status, and escalation endpoints.
- `store_qa.csv`: Example CSV for FAQ/document Q&A.

## Getting Started

### 1. Install Requirements

```bash
pip install fastapi uvicorn streamlit langchain langchain-google-genai langchain-community chromadb pydantic requests
```

### 2. Set up Google API Key

Set your Google Generative AI API key as an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or modify the API key in `customer_agent.py`.

### 3. Run the API Server

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

### 4. Launch the Streamlit App

```bash
streamlit run streamlit_customer_service.py
```

The web app will be available at `http://localhost:8501`

### 5. Interact

- Use the web UI to chat, track orders, file complaints, escalate, and ask questions about documents.
- Use the API endpoints for programmatic access.

## Usage Examples

### Chat Interface
- **Order tracking**: "track my order" or "order status"
- **Returns/complaints**: "I want to return" or "file a complaint"
- **Document Q&A**: "load document", "ask about document", "document status", "clear document"
- **General FAQ**: Ask any question about policies, procedures, timeframes
- **Escalation**: "escalate my complaint"

### Document Q&A Commands
- `load document` - Load a new document for Q&A
- `document status` - Check if a document is loaded
- `clear document` - Clear the current document
- Ask any question about the loaded document

## API Endpoints

- `POST /complaints`: Submit a complaint
  ```json
  {
    "id": "complaint_id",
    "order_id": "ORD123",
    "issue": "Product defective"
  }
  ```

- `GET /orders/{order_id}`: Get order status
  - Example: `GET /orders/ORD123`

- `POST /escalations`: Escalate a complaint
  ```json
  {
    "complaint_id": "complaint_id",
    "reason": "Not resolved in time"
  }
  ```

## Document Q&A (RAG)

The system supports loading and querying documents in multiple formats:

- **CSV files**: Customer service data, product information
- **PDF files**: Manuals, policies, documentation
- **TXT files**: Knowledge base articles, FAQs

Load a document via the web interface or programmatically, then ask natural language questions about its content.

## Architecture

The system uses:

- **LangChain**: For LLM integration and document processing
- **Google Generative AI**: For embeddings and chat completions
- **Chroma**: Vector database for document embeddings
- **FastAPI**: REST API backend
- **Streamlit**: Web interface
- **LangGraph**: For conversation flow management

## Sample Data

- Example order IDs: `ORD123`, `ORD456`
- Sample CSV data in `store_qa.csv` for FAQ responses
- In-memory storage for demo purposes

## Notes

- Requires a Google API key for LLM and embeddings functionality
- For production use, replace in-memory storage with persistent databases
- The system maintains conversation state and document context across interactions
- Supports file upload through the Streamlit interface