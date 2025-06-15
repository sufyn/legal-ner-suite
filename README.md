Legal NER Suite: AI-Powered Legal Document Automation & Analytics



🚀 Project Overview
The Legal NER Suite is a comprehensive, AI-powered web application designed to revolutionize legal document processing, crime analysis, and information management for law enforcement and legal professionals in India. Built with Flask and leveraging advanced Natural Language Processing (NLP), it automates key aspects of FIR (First Information Report) creation, provides intelligent IPC/BNS section predictions, and offers data-driven insights into crime trends.

💡 Problem Solved
This project addresses critical challenges in the legal domain:

Inefficient & Error-Prone Documentation: Manually drafting FIRs is time-consuming and prone to human error, leading to inconsistencies and delays.

Complex Legal Classification: Accurately identifying relevant IPC/BNS sections for a crime requires deep legal expertise and can be subjective or incorrect.

Lack of Actionable Crime Insights: Traditional methods struggle to provide real-time, visual data on crime patterns, hindering proactive policing and resource allocation.

By automating these processes, the suite enhances efficiency, improves accuracy, and provides valuable analytical insights.

✨ Key Features
Named Entity Recognition (NER) & FIR Assistant:
Extracts critical entities (Petitioner, Accused, Date, Place, Statute, etc.) from unstructured case descriptions.
Supports text input and voice input (multi-language: English, Hindi, Telugu) for convenience.
Automatically populates and highlights an editable FIR template.

Automatic FIR Creation (PDF):
Generates a structured, downloadable FIR document in PDF format based on extracted and user-confirmed data.

IPC/BNS Section Prediction:
Predicts the overarching "Mega Category" of crime.
Suggests top relevant IPC/BNS sections using NLP similarity for precise legal classification.

Crime Data Analytics:
Provides visualizations (e.g., heatmaps) to represent crime density and trends across geographical areas, aiding strategic decision-making.

🛠️ Technical Stack
Backend: Python 3.9+ (Flask)
NLP/Machine Learning:
scikit-learn (for TF-IDF, LinearSVC)
pandas (for data manipulation)
joblib (for model persistence)
Potentially: spaCy or Hugging Face Transformers for advanced NER (if custom models are fine-tuned)
Frontend: HTML5, CSS3, JavaScript (Vanilla JS, Web Speech API), Bootstrap 5.x
PDF Generation: ReportLab (or similar Python library)
Data Visualization: Matplotlib, Seaborn (for backend plot generation), Folium (for map visualizations)

🚀 Getting Started
Follow these instructions to set up and run the project locally.

Prerequisites
Python 3.9+
pip (Python package installer)
Installation
Clone the repository:

Code snippet

git clone https://github.com/YourUsername/legal-ner-suite.git
cd legal-ner-suite
(Replace YourUsername/legal-ner-suite.git with your actual GitHub repository URL)

Create a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

```bash
pip install -r requirements.txt


*If `requirements.txt` does not exist, you can create it by running:*

```bash
pip freeze > requirements.txt
# Then manually add any missing packages like reportlab, folium if not included by default pip freeze
Model and Data Setup (Important!)
The application relies on pre-trained NLP models and an IPC sections dataset.

For First-Time Run / Development:
The app.py script includes a mechanism to create dummy model files (tfidf_vectorizer_mega_category.pkl, linear_svc_model_mega_category.pkl) and a basic ipc_sections.csv if they are not found. This allows you to run the application immediately for basic testing, but the predictions will not be accurate.

For Accurate Predictions:
You must replace these dummy files with your actual trained models and your comprehensive ipc_sections.csv (containing mega_category, Section, Offense, Description, and ipc_text_for_similarity columns). Place them in the root directory of your project (same as app.py).

Example file structure for models/data:

legal-ner-suite/
├── app.py
├── ipc_sections.csv
├── tfidf_vectorizer_mega_category.pkl
├── linear_svc_model_mega_category.pkl
├── templates/
│   └── predict.html
└── static/
    └── css/
        └── style.css
├── requirements.txt
└── README.md
Running the Application
Ensure your virtual environment is active.
Run the Flask application:
Bash

python app.py
Open your web browser and navigate to:
http://127.0.0.1:5000/ (This will land on the FIR Assistant tab)
http://127.0.0.1:5000/predict_sections (Directly opens the Case Prediction tab)
Note: Remember to change the app.secret_key in app.py for production deployment.

📂 Project Structure
legal-ner-suite/
├── app.py                        # Main Flask application
├── ipc_sections.csv              # Dataset for IPC sections (IPC, Offense, Description, Mega Category)
├── tfidf_vectorizer_mega_category.pkl # Pre-trained TF-IDF Vectorizer for classification/similarity
├── linear_svc_model_mega_category.pkl # Pre-trained LinearSVC model for Mega Category prediction
├── templates/                    # HTML templates
│   └── predict.html              # Main multi-tabbed UI (FIR, Research, Prediction, Analysis)
├── static/                       # Static files (CSS, JS, images)
│   └── css/
│       └── style.css             # Custom CSS
├── requirements.txt              # Project dependencies
└── README.md                     # Project README
🤝 Usage
FIR Assistant: Type or speak an incident description to extract entities and generate an editable FIR template.
Case Prediction: Input an incident description to receive predicted Mega Categories and top relevant IPC/BNS sections.
Legal Research: (Placeholder/Future scope) Input queries to extract keywords and search for similar cases.
Analysis: (Placeholder/Future scope) View crime data visualizations and heatmaps.
🔮 Future Enhancements
Integration with a more robust, fine-tuned NER model (e.g., using spaCy's Legal-NER or custom trained models on a larger Indian legal corpus).
Advanced query functionality for legal research, including semantic search.
Interactive and real-time crime analytics dashboards with filter options.
User authentication and role-based access control.
Integration with official legal databases for real-time IPC/BNS updates.
🤝 Contributing
We welcome contributions! If you'd like to contribute, please follow these steps:

Fork the repository. 2. Create a new branch (git checkout -b feature/your-feature-name).
Make your changes.
Commit your changes (git commit -m 'feat: Add new feature').
Push to the branch (git push origin feature/your-feature-name).
Open a Pull Request.
Please ensure your code adheres to good practices and includes appropriate tests.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
(You'll need to create a LICENSE file in your root directory if you choose the MIT License)