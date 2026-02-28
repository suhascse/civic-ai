from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime
from transformers import pipeline
from PIL import Image
import torch
from torchvision import models, transforms
import os
import uuid
import shutil
from typing import Optional
from pydantic import BaseModel

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(title="CivicAI – Smart Civic Intelligence System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve uploaded images as static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# ─────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./issues.db")

# Render uses PostgreSQL which gives a "postgres://" URL — SQLAlchemy needs "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ── Models ──────────────────────────────────

class Issue(Base):
    __tablename__ = "issues"

    id              = Column(Integer, primary_key=True, index=True)
    ticket_id       = Column(String, unique=True, index=True)   # human-readable e.g. CIV-0042

    # Citizen info
    name            = Column(String)
    phone           = Column(String, index=True)
    email           = Column(String, nullable=True)

    # Location
    ward_number     = Column(String)
    area_name       = Column(String)
    city            = Column(String)
    pincode         = Column(String)
    landmark        = Column(String, nullable=True)

    # Issue details
    issue_type      = Column(String)
    description     = Column(Text)

    # AI outputs
    ai_text_category  = Column(String)
    ai_image_category = Column(String)
    ai_confidence     = Column(Float, nullable=True)

    # Media
    image_path      = Column(String)

    # Priority
    severity        = Column(Integer)
    priority_score  = Column(Float)

    # Workflow
    status          = Column(String, default="Open")
    assigned_to     = Column(String, nullable=True)
    resolution_note = Column(Text, nullable=True)
    resolved_at     = Column(DateTime, nullable=True)

    upvotes         = Column(Integer, default=0)

    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class IssueStatusHistory(Base):
    __tablename__ = "issue_status_history"

    id          = Column(Integer, primary_key=True, index=True)
    issue_id    = Column(Integer)
    old_status  = Column(String)
    new_status  = Column(String)
    note        = Column(Text, nullable=True)
    changed_at  = Column(DateTime, default=datetime.utcnow)


class Comment(Base):
    __tablename__ = "comments"

    id         = Column(Integer, primary_key=True, index=True)
    issue_id   = Column(Integer, index=True)
    author     = Column(String)
    content    = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ── DB Dependency ────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────
#  AI MODELS
# ─────────────────────────────────────────────

print("Loading AI models…")

text_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

TEXT_LABELS = [
    "Road Damage",
    "Garbage & Sanitation",
    "Water Supply & Leakage",
    "Electrical Problem",
    "Street Light Issue",
    "Sewage & Drainage",
    "Public Property Damage",
    "Noise Complaint",
]

image_model = models.mobilenet_v2(pretrained=True)
image_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Map ImageNet class indices to rough civic categories
IMAGENET_CIVIC_MAP = {
    range(480, 486): "Road Damage",
    range(670, 680): "Garbage & Sanitation",
    range(897, 900): "Water Supply & Leakage",
    range(555, 560): "Electrical Problem",
}

print("AI Models Ready ✅")


# ─────────────────────────────────────────────
#  UTILS
# ─────────────────────────────────────────────

def classify_text(text: str):
    result = text_classifier(text, TEXT_LABELS)
    label = result["labels"][0]
    confidence = round(result["scores"][0] * 100, 1)
    return label, confidence


def classify_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    tensor = image_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = image_model(tensor)
    _, predicted = torch.max(output, 1)
    cls = predicted.item()
    for rng, label in IMAGENET_CIVIC_MAP.items():
        if cls in rng:
            return label
    return "General Civic Issue"


SEVERITY_MAP = {
    "Road": 5,
    "Electricity": 5,
    "Water": 4,
    "Sewage": 4,
    "Garbage": 3,
    "Street Light": 3,
    "Public Property": 2,
    "Noise": 1,
}

STATUS_FLOW = ["Open", "Under Review", "In Progress", "Resolved", "Closed"]


def get_severity(issue_type: str) -> int:
    for key, val in SEVERITY_MAP.items():
        if key.lower() in issue_type.lower():
            return val
    return 2


def calculate_priority(severity: int, upvotes: int = 0) -> float:
    return round(severity * 0.7 + upvotes * 0.3, 2)


def generate_ticket_id(db: Session) -> str:
    count = db.query(func.count(Issue.id)).scalar() or 0
    return f"CIV-{str(count + 1).zfill(4)}"


# ─────────────────────────────────────────────
#  ROUTES – ISSUE SUBMISSION
# ─────────────────────────────────────────────

@app.post("/report-with-ai")
async def report_with_ai(
    name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(""),
    ward_number: str = Form(...),
    area_name: str = Form(...),
    city: str = Form(...),
    pincode: str = Form(...),
    landmark: str = Form(""),
    issue_type: str = Form(...),
    description: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Save image with unique filename
    ext = os.path.splitext(image.filename)[-1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_name)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # AI Classification
    ai_text_category, ai_confidence = classify_text(description)
    ai_image_category = classify_image(image_path)

    severity = get_severity(issue_type)
    priority = calculate_priority(severity)
    ticket_id = generate_ticket_id(db)

    new_issue = Issue(
        ticket_id=ticket_id,
        name=name,
        phone=phone.strip(),
        email=email or None,
        ward_number=ward_number,
        area_name=area_name,
        city=city,
        pincode=pincode,
        landmark=landmark or None,
        issue_type=issue_type,
        description=description,
        ai_text_category=ai_text_category,
        ai_image_category=ai_image_category,
        ai_confidence=ai_confidence,
        image_path=image_path,
        severity=severity,
        priority_score=priority,
    )

    db.add(new_issue)
    db.commit()
    db.refresh(new_issue)

    db.add(IssueStatusHistory(
        issue_id=new_issue.id,
        old_status="None",
        new_status="Open",
        note="Issue submitted by citizen.",
    ))
    db.commit()

    return {
        "message": "Issue submitted successfully",
        "ticket_id": ticket_id,
        "ai_text_category": ai_text_category,
        "ai_image_category": ai_image_category,
        "severity": severity,
        "priority_score": priority,
    }


# ─────────────────────────────────────────────
#  ROUTES – TRACKING
# ─────────────────────────────────────────────

def _serialize_issue(issue: Issue) -> dict:
    return {
        "id": issue.id,
        "ticket_id": issue.ticket_id,
        "name": issue.name,
        "phone": issue.phone,
        "email": issue.email,
        "ward_number": issue.ward_number,
        "area_name": issue.area_name,
        "city": issue.city,
        "pincode": issue.pincode,
        "landmark": issue.landmark,
        "issue_type": issue.issue_type,
        "description": issue.description,
        "ai_text_category": issue.ai_text_category,
        "ai_image_category": issue.ai_image_category,
        "ai_confidence": issue.ai_confidence,
        "image_path": "/" + issue.image_path.replace("\\", "/"),
        "severity": issue.severity,
        "priority_score": issue.priority_score,
        "status": issue.status,
        "assigned_to": issue.assigned_to,
        "resolution_note": issue.resolution_note,
        "resolved_at": issue.resolved_at.isoformat() if issue.resolved_at else None,
        "upvotes": issue.upvotes,
        "created_at": issue.created_at.isoformat() if issue.created_at else None,
        "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
    }


@app.get("/track/{phone}")
def track_by_phone(phone: str, db: Session = Depends(get_db)):
    issues = db.query(Issue).filter(Issue.phone == phone.strip()).order_by(Issue.created_at.desc()).all()
    return [_serialize_issue(i) for i in issues]


@app.get("/ticket/{ticket_id}")
def get_by_ticket(ticket_id: str, db: Session = Depends(get_db)):
    issue = db.query(Issue).filter(Issue.ticket_id == ticket_id.upper()).first()
    if not issue:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return _serialize_issue(issue)


@app.get("/history/{issue_id}")
def get_history(issue_id: int, db: Session = Depends(get_db)):
    history = db.query(IssueStatusHistory)\
        .filter(IssueStatusHistory.issue_id == issue_id)\
        .order_by(IssueStatusHistory.changed_at.asc()).all()
    return [
        {
            "old_status": h.old_status,
            "new_status": h.new_status,
            "note": h.note,
            "changed_at": h.changed_at.isoformat(),
        }
        for h in history
    ]


# ─────────────────────────────────────────────
#  ROUTES – UPVOTE
# ─────────────────────────────────────────────

@app.post("/upvote/{issue_id}")
def upvote(issue_id: int, db: Session = Depends(get_db)):
    issue = db.query(Issue).filter(Issue.id == issue_id).first()
    if not issue:
        raise HTTPException(status_code=404, detail="Issue not found")
    issue.upvotes = (issue.upvotes or 0) + 1
    issue.priority_score = calculate_priority(issue.severity, issue.upvotes)
    db.commit()
    return {"upvotes": issue.upvotes, "priority_score": issue.priority_score}


# ─────────────────────────────────────────────
#  ROUTES – STATUS UPDATE (admin)
# ─────────────────────────────────────────────

class StatusUpdate(BaseModel):
    new_status: str
    note: Optional[str] = None
    assigned_to: Optional[str] = None
    resolution_note: Optional[str] = None


@app.patch("/update-status/{issue_id}")
def update_status(issue_id: int, body: StatusUpdate, db: Session = Depends(get_db)):
    issue = db.query(Issue).filter(Issue.id == issue_id).first()
    if not issue:
        raise HTTPException(status_code=404, detail="Issue not found")

    if body.new_status not in STATUS_FLOW:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {STATUS_FLOW}")

    old_status = issue.status
    issue.status = body.new_status
    issue.updated_at = datetime.utcnow()

    if body.assigned_to:
        issue.assigned_to = body.assigned_to
    if body.resolution_note:
        issue.resolution_note = body.resolution_note
    if body.new_status in ("Resolved", "Closed"):
        issue.resolved_at = datetime.utcnow()

    db.add(IssueStatusHistory(
        issue_id=issue.id,
        old_status=old_status,
        new_status=body.new_status,
        note=body.note,
    ))
    db.commit()
    return {"message": "Status updated", "ticket_id": issue.ticket_id, "new_status": body.new_status}


# ─────────────────────────────────────────────
#  ROUTES – COMMENTS
# ─────────────────────────────────────────────

class CommentBody(BaseModel):
    author: str
    content: str


@app.post("/comment/{issue_id}")
def add_comment(issue_id: int, body: CommentBody, db: Session = Depends(get_db)):
    issue = db.query(Issue).filter(Issue.id == issue_id).first()
    if not issue:
        raise HTTPException(status_code=404, detail="Issue not found")
    comment = Comment(issue_id=issue_id, author=body.author, content=body.content)
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return {"id": comment.id, "author": comment.author, "content": comment.content, "created_at": comment.created_at.isoformat()}


@app.get("/comments/{issue_id}")
def get_comments(issue_id: int, db: Session = Depends(get_db)):
    comments = db.query(Comment).filter(Comment.issue_id == issue_id).order_by(Comment.created_at.asc()).all()
    return [{"id": c.id, "author": c.author, "content": c.content, "created_at": c.created_at.isoformat()} for c in comments]


# ─────────────────────────────────────────────
#  ROUTES – PUBLIC DASHBOARD / STATS
# ─────────────────────────────────────────────

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total       = db.query(func.count(Issue.id)).scalar()
    open_cnt    = db.query(func.count(Issue.id)).filter(Issue.status == "Open").scalar()
    progress    = db.query(func.count(Issue.id)).filter(Issue.status == "In Progress").scalar()
    resolved    = db.query(func.count(Issue.id)).filter(Issue.status == "Resolved").scalar()

    by_type = db.query(Issue.issue_type, func.count(Issue.id))\
        .group_by(Issue.issue_type).all()

    by_ward = db.query(Issue.ward_number, func.count(Issue.id))\
        .group_by(Issue.ward_number)\
        .order_by(func.count(Issue.id).desc())\
        .limit(10).all()

    return {
        "total": total,
        "open": open_cnt,
        "in_progress": progress,
        "resolved": resolved,
        "by_type": [{"type": t, "count": c} for t, c in by_type],
        "by_ward": [{"ward": w, "count": c} for w, c in by_ward],
    }


@app.get("/public-issues")
def public_issues(
    city: Optional[str] = None,
    issue_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Issue)
    if city:
        q = q.filter(Issue.city.ilike(f"%{city}%"))
    if issue_type:
        q = q.filter(Issue.issue_type == issue_type)
    if status:
        q = q.filter(Issue.status == status)
    issues = q.order_by(Issue.priority_score.desc(), Issue.created_at.desc()).offset(offset).limit(limit).all()
    return [_serialize_issue(i) for i in issues]


# ─────────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
