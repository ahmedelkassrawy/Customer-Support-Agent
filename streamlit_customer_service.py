import streamlit as st
import uuid
from langchain_core.messages import AIMessage, HumanMessage
from customer_agent import extract_intent, faq, complaint, order_track, escalate, summarizer, rag, Schema, graph
from rag_module import load_document_for_qa, get_document_status, clear_current_document

st.set_page_config(page_title="Customer Service Agent", page_icon="🎧")
st.title("Customer Service Agent")
st.markdown("""
**Available services:**
- Order tracking: 'track my order' or 'order status'
- Returns/complaints: 'I want to return' or 'file a complaint'
- Document Q&A: 'load document', 'ask about document', 'document status', 'clear document'
- General FAQ: ask any question
- Escalation: 'escalate my complaint'

**Document Q&A Commands:**
- 'load document' - Load a new document for Q&A
- 'document status' - Check if a document is loaded
- 'clear document' - Clear the current document
- Ask any question about the loaded document
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "current_complaint_id" not in st.session_state:
    st.session_state.current_complaint_id = None

if "show_document_uploader" not in st.session_state:
    st.session_state.show_document_uploader = False

user_input = st.text_input("You:", key="user_input")

# Show order ID input if user wants to track order or file complaint
if "show_order_input" not in st.session_state:
    st.session_state.show_order_input = False

if user_input and ("track" in user_input.lower() or "order" in user_input.lower() or "complaint" in user_input.lower() or "return" in user_input.lower()):
    st.session_state.show_order_input = True

# Show document uploader if user wants to load a document
if user_input and any(phrase in user_input.lower() for phrase in ["load document", "upload document", "add document", "new document"]):
    st.session_state.show_document_uploader = True

if st.session_state.show_order_input:
    order_id_input = st.text_input("Enter your Order ID:", key="order_id_input", placeholder="e.g., ORD123, ORD456")

if st.session_state.show_document_uploader:
    st.markdown("### 📄 Document Upload")
    doc_path_input = st.text_input("Enter the path to your document (CSV, PDF, or TXT):", key="doc_path_input", placeholder="e.g., /path/to/document.pdf")
    
    # Alternative: File uploader (for uploaded files)
    uploaded_file = st.file_uploader("Or upload a document:", type=['pdf', 'csv', 'txt'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.temp_doc_path = tmp_file.name
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    if st.button("Load Document") and (doc_path_input or uploaded_file):
        # Use uploaded file path or entered path
        doc_path = st.session_state.get('temp_doc_path', doc_path_input) if uploaded_file else doc_path_input
        
        if doc_path:
            result = load_document_for_qa(doc_path)
            if result["success"]:
                st.success(result["message"])
                st.session_state.show_document_uploader = False
            else:
                st.error(result["message"])

if st.button("Send") and user_input.strip():
    # Add user message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Intent extraction
    state = Schema(
        messages=st.session_state.chat_history,
        question=user_input,
        order_id=None,
        status="",
        complaint_id=None,
        escalation_status=None
    )
    state = extract_intent(state)
    
    # Use user-provided order ID or default
    if state["status"] in ["complaint", "track"]:
        if st.session_state.show_order_input and "order_id_input" in st.session_state and st.session_state.order_id_input:
            state["order_id"] = st.session_state.order_id_input
        else:
            state["order_id"] = "ORD123"  # fallback default
    
    # For escalation, use stored complaint ID
    if state["status"] == "escalate":
        state["complaint_id"] = st.session_state.current_complaint_id
    
    # Run the correct node
    if state["status"] == "faq":
        result = faq(state)
    elif state["status"] == "rag":
        result = rag(state)
    elif state["status"] == "complaint":
        result = complaint(state)
    elif state["status"] == "track":
        result = order_track(state)
    elif state["status"] == "escalate":
        result = escalate(state)
    else:
        result = {"messages": state["messages"] + [AIMessage(content="I'm here to help! Please let me know what you need assistance with.")]}
    
    # Add AI response to chat history
    for msg in result["messages"][len(st.session_state.chat_history):]:
        st.session_state.chat_history.append(msg)
    
    # Store complaint ID if a complaint was filed
    if state["status"] == "complaint" and "complaint_id" in result:
        st.session_state.current_complaint_id = result["complaint_id"]
    
    # Reset order input visibility after processing
    if state["status"] in ["complaint", "track"]:
        st.session_state.show_order_input = False

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Assistant:** {msg.content}")

# Sidebar with current status
st.sidebar.title("📊 Session Status")

# Document status
try:
    doc_status = get_document_status()
    if doc_status["is_loaded"]:
        st.sidebar.success(f"📄 **Document Loaded:** {doc_status['current_document']}")
        if st.sidebar.button("🗑️ Clear Document"):
            result = clear_current_document()
            st.sidebar.success(result["message"])
            st.experimental_rerun()
    else:
        st.sidebar.info("📄 **No document loaded**")
except Exception as e:
    st.sidebar.warning("📄 **Document status unavailable**")

# Complaint status
if st.session_state.current_complaint_id:
    st.sidebar.success(f"📝 **Active Complaint:** {st.session_state.current_complaint_id}")
else:
    st.sidebar.info("📝 **No active complaint**")

# Quick actions
st.sidebar.title("🚀 Quick Actions")
if st.sidebar.button("📄 Load New Document"):
    st.session_state.show_document_uploader = True
    st.experimental_rerun()

if st.sidebar.button("📊 Check Document Status"):
    st.session_state.chat_history.append(HumanMessage(content="document status"))
    st.experimental_rerun()

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()
