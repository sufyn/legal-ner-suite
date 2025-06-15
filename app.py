from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import sqlite3
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification
from collections import Counter
import plotly.express as px
import plotly.io as pio
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json
import base64
import datetime
import pandas as pd
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
import uuid
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# Create Flask app
app = Flask(__name__, 
    template_folder='templates',  # Explicitly set template folder
    static_folder='static'        # Add static folder for assets
)

# Configure app
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# class LegalDoc(db.Model):
#     # __tablename__ = 'legal_docs'
#     id = db.Column(db.String, primary_key=True)
#     title = db.Column(db.String)
#     text = db.Column(db.Text)
#     court = db.Column(db.String)

# class NerResult(db.Model):
#     # __tablename__ = 'ner_results'
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     input_text = db.Column(db.Text)
#     entities = db.Column(db.Text)
#     timestamp = db.Column(db.DateTime, server_default=db.func.current_timestamp())

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_db():
    with app.app_context():
        try:
            db.create_all()
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            
            c.execute('''CREATE TABLE IF NOT EXISTS legal_docs (
                id TEXT PRIMARY KEY, title TEXT, text TEXT, court TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS ner_results (
                id TEXT PRIMARY KEY, module TEXT, input_text TEXT, entities TEXT, fir_template TEXT, keywords TEXT, similar_cases TEXT, predictions TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()
            conn.close()
            print("Database initialized successfully!")
        except Exception as e:
            print(f"Error initializing database: {e}")

# Load fine-tuned NER model
try:
    tokenizer = AutoTokenizer.from_pretrained("ner-legal-model")
    model = AutoModelForTokenClassification.from_pretrained("ner-legal-model")
    ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
except Exception as e:
    print(f"Error loading NER model: {str(e)}")
    ner_pipe = None

# Load sentence transformer model
try:
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading embedding model: {str(e)}")
    embedding_model = None

ENTITY_TYPES = [
    "COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER", "DATE",
    "ORG", "GPE", "STATUTE", "PROVISION", "PRECEDENT", "CASE_NUMBER",
    "WITNESS", "OTHER_PERSON"
]

# Predefined options for dropdowns
COURT_OPTIONS = ["N/A", "Supreme Court", "Delhi High Court", "Bombay High Court", "Kerala High Court", "District Court"]
BNS_OPTIONS = ["N/A", "BNS 103 - Murder", "BNS 318 - Cheating", "BNS 111 - Criminal Conspiracy"]
LANGUAGE_OPTIONS = [
    {"code": "en-IN", "name": "English (India)"},
    {"code": "hi-IN", "name": "Hindi"},
    {"code": "ta-IN", "name": "Tamil"},
    {"code": "te-IN", "name": "Telugu"},
    {"code": "kn-IN", "name": "Kannada"},
    {"code": "ml-IN", "name": "Malayalam"}
]



# Generate FIR template
def generate_fir_template(entities, description):
    predicted_sections = predict_sections(description)
    offense = ", ".join([s[0] for s in predicted_sections if s[0].startswith("3")]) or "N/A"
    provision = ", ".join([s[0] for s in predicted_sections if s[0].startswith("4")]) or "N/A"
 
    return {
        "Date": next((e['word'] for e in entities if e['entity_group'] == 'DATE'), "N/A"),
        "Place": next((e['word'] for e in entities if e['entity_group'] == 'GPE'), "N/A"),
        "Accused": next((e['word'] for e in entities if e['entity_group'] == 'RESPONDENT'), "N/A"),
        "Offense": next((e['word'] for e in entities if e['entity_group'] == 'STATUTE'), "N/A"),
        "Provision": next((e['word'] for e in entities if e['entity_group'] == 'PROVISION'), "N/A"),
        "Description": description,
        "Court": next((e['word'] for e in entities if e['entity_group'] == 'COURT'), "N/A"),
        "Petitioner": next((e['word'] for e in entities if e['entity_group'] == 'PETITIONER'), "N/A"),
        "Witness": next((e['word'] for e in entities if e['entity_group'] == 'WITNESS'), "N/A"),
        "CaseNumber": next((e['word'] for e in entities if e['entity_group'] == 'CASE_NUMBER'), "N/A"),
        "Precedent": next((e['word'] for e in entities if e['entity_group'] == 'PRECEDENT'), "N/A"),
        "Judge": next((e['word'] for e in entities if e['entity_group'] == 'JUDGE'), "N/A"),
        "Lawyer": next((e['word'] for e in entities if e['entity_group'] == 'LAWYER'), "N/A"),
        "Organization": next((e['word'] for e in entities if e['entity_group'] == 'ORG'), "N/A"),
        "OtherPerson": next((e['word'] for e in entities if e['entity_group'] == 'OTHER_PERSON'), "N/A"),
        "Address": "N/A",
        # "PredictedSections": predicted_sections
    }

# Generate official FIR PDF
def generate_fir_pdf(template, filename="fir_template.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    elements = []

    # Custom styles
    title_style = ParagraphStyle(name='Title', parent=styles['Title'], fontName='Times-Bold', fontSize=16, alignment=1)
    heading_style = ParagraphStyle(name='Heading', parent=styles['Heading2'], fontName='Times-Bold', fontSize=12)
    normal_style = ParagraphStyle(name='Normal', parent=styles['Normal'], fontName='Times-Roman', fontSize=11, leading=14)

    # Header
    elements.append(Paragraph("FIRST INFORMATION REPORT", title_style))
    elements.append(Paragraph("(Under Section 154 Cr.P.C.)", normal_style))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"Police Station: [To be filled by Officer]", normal_style))
    elements.append(Paragraph(f"FIR No.: [To be filled by Officer]", normal_style))
    elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%d/%m/%Y')}", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Complainant Details
    elements.append(Paragraph("1. Complainant Details", heading_style))
    complainant_data = [
        ["Name", template["Petitioner"]],
        ["Address", "N/A"],
        ["Organization", template["Organization"]]
    ]
    complainant_table = Table(complainant_data, colWidths=[2 * inch, 4 * inch])
    complainant_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Times-Roman', 11),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(complainant_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Incident Details
    elements.append(Paragraph("2. Incident Details", heading_style))
    incident_data = [
        ["Date of Incident", template["Date"]],
        ["Place of Incident", template["Place"]],
        ["Description", ""]
    ]
    incident_table = Table(incident_data, colWidths=[2 * inch, 4 * inch])
    incident_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Times-Roman', 11),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(incident_table)
    # Add description separately to handle long text
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(template["Description"], normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Offense Details
    elements.append(Paragraph("3. Offense Details", heading_style))
    offense_data = [
        ["Accused", template["Accused"]],
        ["Offense/Statute", template["Offense"]],
        ["Provision", template["Provision"]],
        ["Witness", template["Witness"]],
        ["OtherPerson", template["OtherPerson"]]
    ]
    offense_table = Table(offense_data, colWidths=[2 * inch, 4 * inch])
    offense_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Times-Roman', 11),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(offense_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Legal Context
    elements.append(Paragraph("4. Legal Context", heading_style))
    legal_data = [
        ["Court", template["Court"]],
        ["CaseNumber", template["CaseNumber"]],
        ["Precedent", template["Precedent"]],
        ["Judge", template["Judge"]],
        ["Lawyer", template["Lawyer"]]
    ]
    legal_table = Table(legal_data, colWidths=[2 * inch, 4 * inch])
    legal_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Times-Roman', 11),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ]))
    elements.append(legal_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Signature
    elements.append(Paragraph("5. Signatures", heading_style))
    elements.append(Paragraph("Complainant: _____________________________", normal_style))
    elements.append(Paragraph("Receiving Officer: _____________________________", normal_style))

    doc.build(elements)
    return filename

# Highlight entities in text
def highlight_entities(text, entities):
    highlighted = text
    colors = {
        "COURT": "#36A2EB", "PETITIONER": "#FF6384", "RESPONDENT": "#FFCE56",
        "STATUTE": "#4BC0C0", "DATE": "#9966FF", "GPE": "#FF9F40",
        "PROVISION": "#2ECC71", "WITNESS": "#E74C3C", "CASE_NUMBER": "#9B59B6",
        "PRECEDENT": "#3498DB", "JUDGE": "#E67E22", "LAWYER": "#1ABC9C",
        "ORG": "#34495E", "OTHER_PERSON": "#F1C40F"
    }
    for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
        if entity['entity_group'] in colors:
            word = entity['word']
            color = colors[entity['entity_group']]
            highlighted = highlighted[:entity['start']] + f'<span style="background-color:{color};color:white;padding:2px;">{word}</span>' + highlighted[entity['end']:]
    return highlighted


# Load IPC and BNS datasets and train Random Forest models
def load_legal_datasets():
    # Simulated IPC data (replace with actual ipc.csv)
    ipc_data = pd.DataFrame({
        "Description": ["Cheating and dishonestly inducing delivery of property", "Theft in a dwelling house"],
        "Offense": ["Cheating", "Theft"],
        "Punishment": ["Imprisonment for 7 years and fine", "Imprisonment for 7 years or fine or both"],
        "Section": ["420", "380"]
    })
    # Simulated BNS data (replace with actual bns.csv)
    bns_data = pd.DataFrame({
        "Chapter": ["V", "V"],
        "Chapter_name": ["Of Offences Against Property", "Of Offences Against Property"],
        "Chapter_subtype": ["Cheating", "Theft"],
        "Section": ["318", "303"],
        "Section_name": ["Cheating", "Theft"],
        "Description": ["Cheating and dishonestly inducing delivery of property", "Theft in a dwelling house"]
    })
    all_sections = list(ipc_data["Section"]) + list(bns_data["Section"])
    section_details = {
        "420": {"name": "Cheating", "punishment": "Imprisonment for 7 years and fine"},
        "380": {"name": "Theft", "punishment": "Imprisonment for 7 years or fine or both"},
        "318": {"name": "Cheating", "punishment": "Up to 7 Years Imprisonment + Fine"},
        "303": {"name": "Theft", "punishment": "Up to 7 Years Imprisonment + Fine"}
    }
    
    # Train Random Forest for BNS
    bns_mlb = MultiLabelBinarizer(classes=bns_data["Section"])
    bns_labels = bns_mlb.fit_transform([[s] for s in bns_data["Section"]])
    bns_pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, stop_words='english'),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    bns_pipeline.fit(bns_data["Description"], bns_labels)
    
    # Train Random Forest for IPC
    ipc_mlb = MultiLabelBinarizer(classes=ipc_data["Section"])
    ipc_labels = ipc_mlb.fit_transform([[s] for s in ipc_data["Section"]])
    ipc_pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, stop_words='english'),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    ipc_pipeline.fit(ipc_data["Description"], ipc_labels)
    
    return ipc_data, bns_data, all_sections, section_details, bns_pipeline, bns_mlb, ipc_pipeline, ipc_mlb

ipc_data, bns_data, ALL_SECTIONS, SECTION_DETAILS, bns_pipeline, bns_mlb, ipc_pipeline, ipc_mlb = load_legal_datasets()

# Legal document corpus
legal_docs = [
    {"id": "case1", "title": "Bankruptcy Law Dispute", "text": "The appellant filed a petition under Bankruptcy Code...", "court": "Supreme Court"},
    {"id": "case2", "title": "Criminal Assault Case", "text": "On 5th March, the accused was charged with assault...", "court": "High Court"},
    {"id": "d79fb7f965a74e418212458285c7c213", "title": "Kerala High Court Case", "text": "High Court Of Kerala At Ernakulam... T.R.Ajayan vs M.Ravindran and Nirmala Dinesh...", "court": "High Court"},
    {"id": "90d9a97c7b7749ec8a4f460fda6f937e", "title": "Tax Dispute Case", "text": "Hongkong Bank account... related to loan from broker, Rahul & Co...", "court": "District Court"},
    {"id": "a325c57ba5b84c6fa46bee65e6616633", "title": "Punjab-Haryana Criminal Case", "text": "Agya Kaur, mother-in-law of the deceased lived separately from Tarlochan Singh...", "court": "High Court"}
]

# Build FAISS index
def build_faiss_index(docs):
    if embedding_model is None:
        return None, None
    texts = [doc['text'] for doc in docs]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

faiss_index, doc_embeddings = build_faiss_index(legal_docs)

# Search similar cases
def search_similar_cases(query, top_k=3, court_filter="All"):
    if embedding_model is None or faiss_index is None:
        return []
    filtered_docs = [doc for doc in legal_docs if court_filter == "All" or doc["court"] == court_filter]
    if not filtered_docs:
        return []
    temp_index, temp_embeddings = build_faiss_index(filtered_docs)
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = temp_index.search(query_embedding, min(top_k, len(filtered_docs)))
    return [filtered_docs[idx] for idx in I[0]]

# Predict BNS/IPC sections
# def predict_sections(text):
#     bns_probs = bns_pipeline.predict_proba([text])[0]
#     bns_sections = [(bns_mlb.classes_[i], prob) for i, prob in enumerate(bns_probs) if prob > 0.5]
#     ipc_probs = ipc_pipeline.predict_proba([text])[0]
#     ipc_sections = [(ipc_mlb.classes_[i], prob) for i, prob in enumerate(ipc_probs) if prob > 0.5]
#     return bns_sections + ipc_sections

def predict_sections(text):
    # Get probabilities for BNS
    bns_probs = bns_pipeline.predict_proba([text])
    if hasattr(bns_probs, "toarray"):
        bns_probs = bns_probs.toarray()
    bns_probs = bns_probs[0] if hasattr(bns_probs, "__getitem__") else bns_probs

    bns_sections = []
    for i, prob in enumerate(bns_probs):
        # If prob is an array, flatten and take the first scalar
        if hasattr(prob, "__len__") and not isinstance(prob, str):
            prob = prob.flatten()[0] if hasattr(prob, "flatten") else prob[0]
        prob = float(prob)
        if prob > 0.5:
            bns_sections.append((str(bns_mlb.classes_[i]), prob))

    # Get probabilities for IPC
    ipc_probs = ipc_pipeline.predict_proba([text])
    if hasattr(ipc_probs, "toarray"):
        ipc_probs = ipc_probs.toarray()
    ipc_probs = ipc_probs[0] if hasattr(ipc_probs, "__getitem__") else ipc_probs

    ipc_sections = []
    for i, prob in enumerate(ipc_probs):
        if hasattr(prob, "__len__") and not isinstance(prob, str):
            prob = prob.flatten()[0] if hasattr(prob, "flatten") else prob[0]
        prob = float(prob)
        if prob > 0.5:
            ipc_sections.append((str(ipc_mlb.classes_[i]), prob))

    return bns_sections + ipc_sections

@app.route('/')
def index():
    return render_template('index.html')

# if user  is login then redirect to home page
# @app.before_request
# def before_request():
#     if current_user.is_authenticated:
#         return redirect(url_for('home'))
    
# @app.route('/home')
# def home():
#     return render_template('ner.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

            if not all([username, email, password]):
                flash('All fields are required')
                return redirect(url_for('signup'))

            if User.query.filter_by(username=username).first():
                flash('Username already exists')
                return redirect(url_for('signup'))
            
            if User.query.filter_by(email=email).first():
                flash('Email already registered')
                return redirect(url_for('signup'))

            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.')
            return redirect(url_for('signup'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')

            if not all([username, password]):
                flash('All fields are required')
                return redirect(url_for('login'))

            user = User.query.filter_by(username=username).first()

            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for('ner'))
            
            flash('Invalid username or password')
        except Exception as e:
            flash('An error occurred during login. Please try again.')
        
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/predict')
@login_required
def predict():
    return render_template('predict.html')

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500


@app.route("/ner", methods=["GET", "POST"])
def ner():
    active_tab = request.form.get("active_tab", "fir")
    error = None
    results = []
    fir_template = None
    highlighted_text = None
    keywords = None
    similar_cases = None
    predictions = None
    plot_html = None
    json_href = None
    pdf_href = None
    record_id = str(uuid.uuid4())

    if request.method == "POST":
        if not ner_pipe:
            error = "NER model not loaded."
        else:
            input_text = request.form.get("description")
            court_filter = request.form.get("court_filter", "All")

            if active_tab == "fir":
                input_type = request.form.get("input_type")
                if input_type == "text":
                    if not input_text:
                        error = "Please enter a case description."
                    else:
                        results = ner_pipe(input_text)
                        fir_template = generate_fir_template(results, input_text)
                        highlighted_text = highlight_entities(input_text, results)
                elif input_type == "file":
                    file = request.files.get("file")
                    if not file or not file.filename.endswith(".json"):
                        error = "Please upload a valid JSON file."
                    else:
                        data = json.load(file)
                        input_text = json.dumps(data)
                        for item in data:
                            if "text" in item:
                                item_results = ner_pipe(item["text"])
                                results.extend(item_results)
                                item["entities"] = item_results
                        fir_template = generate_fir_template(results, input_text)

                if results:
                    import numpy as np

                    def convert_to_builtin_type(obj):
                        if isinstance(obj, dict):
                            return {k: convert_to_builtin_type(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_builtin_type(i) for i in obj]
                        elif isinstance(obj, np.generic):
                            return obj.item()
                        return obj

                    # Filter valid entities
                    valid_results = [r for r in results if r['entity_group'] in ENTITY_TYPES]
                    valid_results = convert_to_builtin_type(valid_results)

                    # valid_results = [r for r in results if r['entity_group'] in ENTITY_TYPES]
                    if fir_template:
                        pdf_file = generate_fir_pdf(fir_template)
                        with open(pdf_file, "rb") as f:
                            b64_pdf = base64.b64encode(f.read()).decode()
                        pdf_href = f'data:application/pdf;base64,{b64_pdf}'
                    json_str = json.dumps(valid_results)
                    b64_json = base64.b64encode(json_str.encode()).decode()
                    json_href = f'data:application/json;base64,{b64_json}'
                    entity_counts = Counter([r['entity_group'] for r in valid_results])
                    if entity_counts:
                        df = pd.DataFrame({"Entity": entity_counts.keys(), "Count": entity_counts.values()})
                        fig = px.bar(df, x="Entity", y="Count", title="Entity Distribution", color="Entity")
                        plot_html = pio.to_html(fig, full_html=False)

                    conn = sqlite3.connect("users.db")
                    c = conn.cursor()
                    c.execute("INSERT INTO ner_results (id, module, input_text, entities, fir_template) VALUES (?, ?, ?, ?, ?)",
                              (record_id, "fir", input_text, json.dumps(valid_results), json.dumps(fir_template)))
                    conn.commit()
                    conn.close()

            elif active_tab == "research":
                if not input_text:
                    error = "Please enter a case or query."
                else:
                    results = ner_pipe(input_text)
                    keywords = [r["word"] for r in results if r["entity_group"] in ["STATUTE", "PROVISION", "PRECEDENT", "ORG", "COURT", "CASE_NUMBER", "GPE"]]
                    similar_cases = search_similar_cases(input_text, court_filter=court_filter)
                    if keywords:
                        keyword_labels = [r['entity_group'] for r in results if r['word'] in keywords]
                        label_counts = Counter(keyword_labels)
                        df = pd.DataFrame({"Label": label_counts.keys(), "Count": label_counts.values()})
                        fig = px.pie(df, names="Label", values="Count", title="Keyword Label Distribution")
                        plot_html = pio.to_html(fig, full_html=False)
                    json_str = json.dumps({"keywords": keywords, "similar_cases": similar_cases})
                    b64_json = base64.b64encode(json_str.encode()).decode()
                    json_href = f'data:application/json;base64,{b64_json}'

                    conn = sqlite3.connect("users.db")
                    c = conn.cursor()
                    c.execute("INSERT INTO ner_results (id, module, input_text, keywords, similar_cases) VALUES (?, ?, ?, ?, ?)",
                              (record_id, "research", input_text, json.dumps(keywords), json.dumps(similar_cases)))
                    conn.commit()
                    conn.close()
                    
            elif active_tab == "analysis":
                return render_template("heatmap.html")


            elif active_tab == "predict":
                if not input_text:
                    error = "Please enter an incident description."
                else:
                    results = ner_pipe(input_text)
                    predicted_sections = predict_sections(input_text)
                    predictions = [
                        {"section": s[0], "name": SECTION_DETAILS.get(s[0], {}).get("name", "Unknown"),
                         "punishment": SECTION_DETAILS.get(s[0], {}).get("punishment", "Unknown"), "score": s[1]}
                        for s in predicted_sections
                    ]
                    json_str = json.dumps({"input": input_text, "predictions": predictions})
                    b64_json = base64.b64encode(json_str.encode()).decode()
                    json_href = f'data:application/json;base64,{b64_json}'

                    conn = sqlite3.connect("users.db")
                    c = conn.cursor()
                    c.execute("INSERT INTO ner_results (id, module, input_text, predictions) VALUES (?, ?, ?, ?)",
                              (record_id, "predict", input_text, json.dumps(predictions)))
                    conn.commit()
                    conn.close()

    return render_template("ner.html", 
                         active_tab=active_tab,
                         error=error,
                         results=results,
                         fir_template=fir_template,
                         highlighted_text=highlighted_text,
                         keywords=keywords,
                         similar_cases=similar_cases,
                         predictions=predictions,
                         plot_html=plot_html,
                         json_href=json_href,
                         pdf_href=pdf_href,
                         court_options=COURT_OPTIONS,
                         section_options=ALL_SECTIONS,
                         language_options=LANGUAGE_OPTIONS,
                         record_id=record_id)


@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    return render_template("heatmap.html", active_tab="analysis")

@app.route("/finalize_fir", methods=["POST"])
def finalize_fir():
    record_id = request.form.get("record_id")
    if not record_id:
        return render_template("ner.html", active_tab="fir", error="Invalid session. Please start over.", 
                             court_options=COURT_OPTIONS, section_options=ALL_SECTIONS, language_options=LANGUAGE_OPTIONS)

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT entities, input_text, fir_template FROM ner_results WHERE id = ?", (record_id,))
    result = c.fetchone()
    if not result:
        conn.close()
        return render_template("ner.html", active_tab="fir", error="Record not found.", 
                             court_options=COURT_OPTIONS, section_options=ALL_SECTIONS, language_options=LANGUAGE_OPTIONS)

    valid_results = json.loads(result[0])
    input_text = result[1]
    old_fir_template = json.loads(result[2])
    conn.close()

    fir_template = {
        "Petitioner": request.form.get("petitioner", old_fir_template["Petitioner"]),
        "Address": request.form.get("address", old_fir_template.get("Address", "N/A")),
        "Organization": request.form.get("organization", old_fir_template["Organization"]),
        "Date": request.form.get("date", old_fir_template["Date"]),
        "Place": request.form.get("place", old_fir_template["Place"]),
        "Description": request.form.get("description", old_fir_template["Description"]),
        "Accused": request.form.get("accused", old_fir_template["Accused"]),
        "Offense": request.form.get("offense", old_fir_template["Offense"]),
        "Provision": request.form.get("provision", old_fir_template["Provision"]),
        "Witness": request.form.get("witness", old_fir_template["Witness"]),
        "OtherPerson": request.form.get("otherperson", old_fir_template["OtherPerson"]),
        "Court": request.form.get("court", old_fir_template["Court"]),
        "CaseNumber": request.form.get("casenumber", old_fir_template["CaseNumber"]),
        "Precedent": request.form.get("precedent", old_fir_template["Precedent"]),
        "Judge": request.form.get("judge", old_fir_template["Judge"]),
        "Lawyer": request.form.get("lawyer", old_fir_template["Lawyer"]),
        "PredictedSections": old_fir_template["PredictedSections"]
    }

    if valid_results:
        entity_counts = Counter([r['entity_group'] for r in valid_results])
        df = pd.DataFrame({"Entity": entity_counts.keys(), "Count": entity_counts.values()})
        fig = px.bar(df, x="Entity", y="Count", title="Entity Distribution", color="Entity")
        plot_html = pio.to_html(fig, full_html=False)
    else:
        plot_html = None

    pdf_file = generate_fir_pdf(fir_template)
    with open(pdf_file, "rb") as f:
        b64_pdf = base64.b64encode(f.read()).decode()
    pdf_href = f'data:application/pdf;base64,{b64_pdf}'

    json_str = json.dumps(valid_results)
    b64_json = base64.b64encode(json_str.encode()).decode()
    json_href = f'data:application/json;base64,{b64_json}'

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE ner_results SET input_text = ?, entities = ?, fir_template = ? WHERE id = ?",
              (fir_template["Description"], json.dumps(valid_results), json.dumps(fir_template), record_id))
    conn.commit()
    conn.close()

    return render_template("results.html", 
                         results=valid_results, 
                         fir_template=fir_template, 
                         plot_html=plot_html, 
                         highlighted_text=highlight_entities(input_text, valid_results),
                         json_href=json_href,
                         pdf_href=pdf_href)


@app.route("/results", methods=["GET"])
def results():
    return render_template("results.html", 
                         results=None, 
                         fir_template=None, 
                         plot_html=None, 
                         highlighted_text=None,
                         json_href=None,
                         pdf_href=None)

# @app.route("/search", methods=["GET", "POST"])
# def search():
#     if request.method == "POST":
#         query = request.form["query"]
#         results = search_db(query)
#         return render_template("search_results.html", results=results)
#     else:
#         return render_template("search.html")
    

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Initialize database
    init_db()
    
    # Run the app
    app.run(debug=True) 