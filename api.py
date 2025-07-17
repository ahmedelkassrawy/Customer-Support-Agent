from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4

app = FastAPI()

# In-memory storage
complaints = {}
orders = {
    "ORD123": {"status": "Shipped", "estimated_delivery": "2025-07-20"},
    "ORD456": {"status": "Processing", "estimated_delivery": "2025-07-25"}
}
escalations = []

class Complaint(BaseModel):
    id: str
    order_id : str
    issue : str

class Escalation(BaseModel):
    id: Optional[str] = None
    complaint_id: str
    reason: str

@app.post("/complaints")
def create_complaint(complaint: Complaint):
    if complaint.id in complaints:
        raise HTTPException(status_code=400, detail="Complaint already exists")
    
    complaints[complaint.id] = complaint
    
    return {"message": "Complaint created successfully", "complaint_id": complaint.id}

@app.get("/orders/{order_id}")
def get_order_status(order_id: str):
    if order_id not in orders:
        raise HTTPException(status_code=404, detail="Order not found")
    
    return {"order": orders[order_id]}

@app.post("/escalations")
def escalate(escalation : Escalation):
    if escalation.complaint_id not in complaints:
        raise HTTPException(status_code=404, detail="Complaint not found")
    
    escalation.id = str(uuid4())
    escalations.append(escalation)

    return {"message": "Escalation created successfully", "escalation_id": escalation.id}