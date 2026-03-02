🧠 CivicAI – Smart Civic Intelligence System

AI-Powered Civic Issue Reporting & Management Platform :

CivicAI is a full-stack AI-powered civic complaint management system that allows citizens to report infrastructure and public service issues, automatically classifies them using Artificial Intelligence, prioritizes them intelligently, and enables real-time tracking and transparency.

🚀 Live Features : 
👤 Citizen Features
📸 Submit complaints with image upload
📝 Detailed issue description
🧠 AI-powered automatic text classification
🖼️ AI-powered image validation using CNN
🏷️ Auto-generated Ticket ID (e.g., CIV-0042)
📍 Ward-based routing
⚡ Smart priority scoring
🔼 Community upvoting system
🔎 Track issues by:
Phone number
Ticket ID

📜 View complete status timeline

💬 Add comments to complaints

📊 Public dashboard with live statistics

🤖 AI Capabilities : 
1️⃣ Text Classification (NLP)

Model: facebook/bart-large-mnli

Zero-shot classification

Categories include:

Road Damage

Garbage & Sanitation

Water Supply & Leakage

Electrical Problem

Street Light Issue

Sewage & Drainage

Public Property Damage

Noise Complaint

2️⃣ Image Classification (Computer Vision)

Model: MobileNet V2 (pretrained on ImageNet)

Image validation mapped to civic categories

Cross-verifies complaint authenticity

3️⃣ Confidence Scoring

AI returns confidence percentage for text classification

🏗️ System Architecture
Frontend : 
Custom responsive UI
Vanilla JavaScript
Multi-page SPA-like behavior
Real-time dashboard
Modal-based issue details
Drag-and-drop image upload
Backend : 
FastAPI
SQLAlchemy ORM
SQLite (local) / PostgreSQL (production-ready)
RESTful API
CORS enabled
Static image serving

🗄️ Database Models
Issue

ticket_id

citizen details (name, phone, email)

location (ward, area, city, pincode, landmark)

issue_type

description

ai_text_category

ai_image_category

ai_confidence

severity

priority_score

status

assigned_to

resolution_note

upvotes

timestamps

IssueStatusHistory

old_status

new_status

note

changed_at

Comment

author

content

created_at

📊 Smart Prioritization Logic

Priority Score =

(severity × 0.7) + (upvotes × 0.3)

Severity mapping:

Issue Type	Severity
Road	5
Electricity	5
Water	4
Sewage	4
Garbage	3
Street Light	3
Public Property	2
Noise	1
🔄 Complaint Workflow
Open → Under Review → In Progress → Resolved → Closed

Every status change is:

Logged in IssueStatusHistory

Timestamped

Auditable

🌐 API Endpoints
Issue Submission
POST /report-with-ai
Tracking
GET /track/{phone}
GET /ticket/{ticket_id}
GET /history/{issue_id}
Community
POST /upvote/{issue_id}
POST /comment/{issue_id}
GET  /comments/{issue_id}
Admin
PATCH /update-status/{issue_id}
Dashboard
GET /stats
GET /public-issues
Health Check
GET /health
📦 Installation & Setup
1️⃣ Clone Repository
git clone <your-repo-url>
cd civicai
2️⃣ Install Dependencies
python -m pip install fastapi uvicorn sqlalchemy transformers torch torchvision pillow python-multipart
3️⃣ Run Application
python -m uvicorn main:app --reload

Open:

http://localhost:8000
🧠 AI Model Download Note

On first run, models will automatically download:

facebook/bart-large-mnli

MobileNet V2

Internet connection required initially.

📊 Dashboard Features

Total Issues

Open Issues

In Progress

Resolved

Issues by Type (bar chart)

Top Wards by Volume

Recent Issues list

🔐 Security Considerations

CORS enabled (can be restricted in production)

Environment variable support for DATABASE_URL

PostgreSQL-ready deployment

Unique ticket IDs

Image stored securely with UUID filenames

🚀 Deployment Ready

Supports:

SQLite (local)

PostgreSQL (Render / Railway / AWS)

Static file serving for uploads

Environment variable configuration

💡 Innovation Highlights (Hackathon Pitch Points)

Multimodal AI (Text + Image)

Zero-shot NLP classification

AI confidence scoring

Smart priority algorithm

Community-driven upvoting

Transparent issue lifecycle tracking

Ward-level analytics

Public dashboard transparency

Production-ready backend architecture

🏆 Why CivicAI Is Different

Most civic complaint systems are:

Manual categorization

Static forms

No intelligence

No prioritization

CivicAI adds:

AI-driven categorization

Smart prioritization

Transparency & analytics

Community engagement

End-to-end issue lifecycle tracking