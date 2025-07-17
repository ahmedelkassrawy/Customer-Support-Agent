# Imports
import os
import uuid
import requests
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Optional
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain import hub
from rag_module import load_document_for_qa, ask_document_question, get_document_status, clear_current_document
os.environ["GOOGLE_API_KEY"] = "AIzaSyAib5iH_zllA8RKdTZLIzKc9T0ajbpm-ic"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

class Schema(MessagesState):
    question: str
    status: str
    escalation_status: Optional[str] = None
    order_id: Optional[str] = None
    complaint_id: Optional[str] = None

class RAG(BaseModel):
    """RAG model for customer service."""
    doc_path: str
    question: str

# Extract intent from latest message
def extract_intent(state: Schema):
    last_message = state["messages"][-1].content.lower()
    
    # More specific intent detection with priority order
    if ("complaint" in last_message or "complain" in last_message) and ("submit" in last_message or "file" in last_message):
        state["status"] = "complaint"
    elif ("return" in last_message or "exchange" in last_message) and ("want" in last_message or "can i" in last_message or "how to" in last_message or "need to" in last_message):
        state["status"] = "complaint"
    elif ("track" in last_message or "status" in last_message) and ("order" in last_message):
        state["status"] = "track"
    elif "escalate" in last_message or "escalation" in last_message:
        state["status"] = "escalate"
    # RAG-specific questions - when user mentions document queries or wants to ask about document content
    elif any(word in last_message for word in ["document", "file", "pdf", "csv", "txt", "upload", "load"]) or "document" in last_message:
        state["status"] = "rag"
    # FAQ questions - questions about policies, procedures, timeframes
    elif any(word in last_message for word in ["how many", "how long", "what is", "when", "where", "why", "policy", "days", "time", "hours"]):
        state["status"] = "faq"
    else:
        state["status"] = "faq"
    
    return state

def rag(state: Schema):
    """RAG function for document Q&A using the separate RAG module."""
    try:
        user_input = state["question"].lower()
        
        # Check if user wants to see document status (most specific first)
        if "document status" in user_input or ("status" in user_input and "document" in user_input):
            status = get_document_status()
            if status["is_loaded"]:
                message = f"📄 Currently loaded document: {status['current_document']}\nYou can ask any questions about this document!"
            else:
                message = "📄 No document is currently loaded. Use 'load document' to load a new document."
            return {"messages": state["messages"] + [AIMessage(content=message)]}
        
        # Check if user wants to clear the document
        elif "clear document" in user_input:
            result = clear_current_document()
            return {"messages": state["messages"] + [AIMessage(content=result["message"])]}
        
        # Check if user wants to load a document (specific phrases only)
        elif any(phrase in user_input for phrase in ["load document", "upload document", "add document", "new document"]) or (("load" in user_input or "upload" in user_input) and any(word in user_input for word in ["file", "pdf", "csv", "txt"])):
            print("📄 Document Loading Mode")
            print("Please provide the path to your document (CSV, PDF, or TXT):")
            doc_path = input("Enter document path: ").strip()
            
            result = load_document_for_qa(doc_path)
            return {"messages": state["messages"] + [AIMessage(content=result["message"])]}
        
        # Otherwise, treat it as a question about the loaded document
        else:
            result = ask_document_question(state["question"])
            
            if result["success"]:
                return {"messages": state["messages"] + [AIMessage(content=result["answer"])]}
            else:
                # If no document is loaded, prompt user to load one
                message = result["message"] + "\n\nTo load a document, say 'load document' or mention a file type (PDF, CSV, TXT)."
                return {"messages": state["messages"] + [AIMessage(content=message)]}
        
    except Exception as e:
        error_message = f"Sorry, I encountered an error: {str(e)}"
        return {"messages": state["messages"] + [AIMessage(content=error_message)]}

# FAQ retriever
def faq(state: Schema):
    try:
        # Load and process the CSV file
        csv_path = "/workspaces/Agent/customer_agent/store_qa.csv"
        loader = CSVLoader(file_path=csv_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        docs = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(docs, embeddings)
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        
        # Get the answer from the QA chain
        question = state["question"]
        response = qa_chain.invoke({"query": question})
        
        # Extract the result from the response
        if isinstance(response, dict) and "result" in response:
            answer = response["result"]
        else:
            answer = str(response)
        
        return {"messages": state["messages"] + [AIMessage(content=answer)]}
        
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        fallback_response = "I'd be happy to help you! I can assist with returns, refunds, shipping, exchanges, coupons, and warranty questions. Please contact customer service at support@company.com for other inquiries."
        return {"messages": state["messages"] + [AIMessage(content=fallback_response)]}

# Submit complaint
def complaint(state: Schema):
    # Check if order_id is provided and valid
    if not state.get("order_id"):
        return {"messages": state["messages"] + [AIMessage(content="Order ID is required to submit a complaint. Please provide a valid order ID.")]}
    
    complaint_id = str(uuid.uuid4())
    
    url = "http://localhost:8000/complaints"
    payload = {
        "id": complaint_id,
        "order_id": str(state["order_id"]),  # Ensure it's a string
        "issue": state["messages"][-1].content
    }
    
    try:
        response = requests.post(url, json = payload)
        print("API Response Status Code:", response.status_code)
        print("API Response Text:", response.text)

        if response.status_code == 200:
            return {
                "messages": state["messages"] + [AIMessage(content=f"Complaint submitted successfully. Complaint ID: {complaint_id}")],
                "complaint_id": complaint_id
            }
        else:
            return {"messages": state["messages"] + [AIMessage(content=f"Error submitting complaint: {response.status_code} - {response.text}")]}
    except requests.exceptions.RequestException as e:
        return {"messages": state["messages"] + [AIMessage(content=f"Error connecting to complaint system: {str(e)}")]}


# Track order
def order_track(state: Schema):
    url = f"http://localhost:8000/orders/{state['order_id']}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            order_data = response.json()
            return {"messages": state["messages"] + [AIMessage(content=f"Order Status: {order_data}")]}
        else:
            return {"messages": state["messages"] + [AIMessage(content=f"Order not found or error: {response.status_code}")]}
    except requests.exceptions.RequestException as e:
        return {"messages": state["messages"] + [AIMessage(content=f"Error connecting to order tracking system: {str(e)}")]}

# Escalate complaint
def escalate(state: Schema):
    if not state.get("complaint_id"):
        return {"messages": state["messages"] + [AIMessage(content="No complaint ID found. Please submit a complaint first.")]}
        
    url = "http://localhost:8000/escalations"
    payload = {
        "complaint_id": state["complaint_id"],
        "reason": state["messages"][-1].content
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return {"messages": state["messages"] + [AIMessage(content="Complaint escalated successfully!")]}
        else:
            return {"messages": state["messages"] + [AIMessage(content=f"Error escalating complaint: {response.status_code}")]}
    except requests.exceptions.RequestException as e:
        return {"messages": state["messages"] + [AIMessage(content=f"Error connecting to escalation system: {str(e)}")]}

# Summarize chat history
def summarizer(state: Schema):
    user_messages = "\n".join([msg.content for msg in state["messages"] if msg.type == "human"])
    summary = llm.invoke(f"Summarize this conversation:\n{user_messages}")
    return {"messages": state["messages"] + [AIMessage(content=summary)]}

# Setup memory checkpointer
checkpointer = MemorySaver()

# LangGraph workflow
workflow = StateGraph(Schema)

# Nodes
workflow.add_node("extract_intent", extract_intent)
workflow.add_node("faq", faq)
workflow.add_node("rag", rag)
workflow.add_node("complaint", complaint)
workflow.add_node("order_track", order_track)
workflow.add_node("escalate", escalate)
workflow.add_node("summarizer", summarizer)

# Edges
workflow.add_edge(START, "extract_intent")
workflow.add_conditional_edges(
    "extract_intent",
    lambda state: state["status"],
    {
        "faq": "faq",
        "rag": "rag",
        "complaint": "complaint",
        "track": "order_track",
        "escalate": "escalate"
    }
)

# Add edges to END
workflow.add_edge("faq", END)
workflow.add_edge("rag", END)
workflow.add_edge("complaint", END)
workflow.add_edge("order_track", END)
workflow.add_edge("escalate", END)

# Compile
graph = workflow.compile(checkpointer=checkpointer)

def run_customer_service():
    """Interactive customer service chat that runs until stopped."""
    print("Customer Service Agent")
    print("=" * 50)
    print("Available services:")
    print("- Order tracking: 'track my order' or 'order status'")
    print("- Returns/complaints: 'I want to return' or 'file a complaint'")
    print("- Document Q&A: 'load document', 'ask about document', 'document status', 'clear document'")
    print("- General FAQ: ask any question")
    print("- Escalation: 'escalate my complaint'")
    print("\nDocument Q&A Commands:")
    print("- 'load document' - Load a new document for Q&A")
    print("- 'document status' - Check if a document is loaded")
    print("- 'clear document' - Clear the current document")
    print("- Ask any question about the loaded document")
    print("\nType 'exit' or 'quit' to stop the chat")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("🤖 Assistant: Thank you for using our customer service! Have a great day!")
                break
            
            if not user_input:
                print("🤖 Assistant: Please enter your question or concern.")
                continue
            
            # Create a fresh thread for each conversation to avoid state pollution
            thread = {"configurable": {"thread_id": f"session_{uuid.uuid4()}"}}
            
            # Get order ID only if the intent clearly requires it
            order_id = None
            user_input_lower = user_input.lower()
            
            # Check if we need order ID based on intent
            needs_order_id = (
                ("track" in user_input_lower and "order" in user_input_lower) or
                ("return" in user_input_lower and ("want" in user_input_lower or "can i" in user_input_lower or "need to" in user_input_lower)) or
                ("complaint" in user_input_lower and ("file" in user_input_lower or "submit" in user_input_lower))
            )
            
            if needs_order_id:
                order_input = input("📦 Order ID (press Enter for ORD123): ").strip()
                order_id = order_input if order_input else "ORD123"
            
            # Create fresh state for this interaction
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "question": user_input,
                "order_id": order_id,
                "status": "",  # Reset status
                "complaint_id": None,  # Reset complaint ID
                "escalation_status": None  # Reset escalation status
            }
            
            print("🤖 Assistant: Processing your request...")
            
            # Run the workflow
            result = graph.invoke(initial_state, config=thread)
            
            # Display the assistant's response
            assistant_responded = False
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"🤖 Assistant: {msg.content}")
                    assistant_responded = True
                    
            if not assistant_responded:
                print("🤖 Assistant: I'm here to help! Please let me know what you need assistance with.")
                    
        except KeyboardInterrupt:
            print("\n\n🤖 Assistant: Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("🤖 Assistant: Sorry, something went wrong. Please try again.")

if __name__ == "__main__":
    run_customer_service()
