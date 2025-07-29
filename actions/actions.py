# Custom Rasa actions for logic
import os
import sqlite3
import requests
import json
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, AllSlotsReset
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Spacer
from dotenv import load_dotenv

# --- Initial Setup ---
load_dotenv()
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Database Setup ---
def init_db():
    """Initializes the SQLite database and the sessions table."""
    conn = sqlite3.connect('career_counsellor.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            name TEXT,
            interests TEXT,
            strengths TEXT,
            subjects TEXT,
            resume_keywords TEXT,
            recommended_career TEXT,
            report_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Career Data (Can be expanded) ---
CAREER_DATA = {
    'Tech / Data Science': {
        'keywords': ['python', 'java', 'data', 'analysis', 'machine learning', 'ai', 'software', 'developer', 'engineer', 'code', 'computer', 'statistics', 'math', 'backend', 'frontend', 'cloud'],
        'description': 'Careers in this field involve designing, developing, and applying technology and data to solve complex problems. Roles include Software Engineer, Data Scientist, AI Specialist, and Cloud Architect.',
        'courses': [
            {'title': 'Google Data Analytics Professional Certificate', 'url': 'https://www.coursera.org/professional-certificates/google-data-analytics'},
            {'title': 'Meta Back-End Developer Professional Certificate', 'url': 'https://www.coursera.org/professional-certificates/meta-back-end-developer'}
        ]
    },
    'Arts / Design': {
        'keywords': ['creative', 'art', 'design', 'music', 'drawing', 'painting', 'visual', 'style', 'photoshop', 'illustrator', 'ui', 'ux', 'figma'],
        'description': 'This domain is for creative individuals who enjoy expressing themselves visually or through performance. Careers include Graphic Designer, UX/UI Designer, Artist, and Animator.',
        'courses': [
            {'title': 'Google UX Design Professional Certificate', 'url': 'https://www.coursera.org/professional-certificates/google-ux-design'},
            {'title': 'CalArts Graphic Design Specialization', 'url': 'https://www.coursera.org/specializations/graphic-design'}
        ]
    },
    'Commerce / Management': {
        'keywords': ['business', 'management', 'finance', 'economics', 'marketing', 'leadership', 'sales', 'trade', 'commerce', 'accounts'],
        'description': 'This field focuses on business operations, finance, and leadership. Potential careers are Business Analyst, Marketing Manager, Financial Advisor, and Entrepreneur.',
        'courses': [
            {'title': 'Introduction to Marketing by Wharton', 'url': 'https://www.coursera.org/learn/wharton-marketing'},
            {'title': 'Financial Markets by Yale', 'url': 'https://www.coursera.org/learn/financial-markets-global'}
        ]
    },
    'Medicine / Biology': {
        'keywords': ['biology', 'chemistry', 'doctor', 'nurse', 'health', 'medical', 'research', 'genetics', 'anatomy', 'patient', 'care'],
        'description': 'For those passionate about health, science, and helping others. Careers range from Doctor and Nurse to Medical Researcher and Pharmacist.',
        'courses': [
            {'title': 'Anatomy Specialization by University of Michigan', 'url': 'https://www.coursera.org/specializations/anatomy'},
            {'title': 'Introduction to the Biology of Cancer', 'url': 'https://www.coursera.org/learn/cancer'}
        ]
    },
    'Law / Social Sciences': {
        'keywords': ['law', 'justice', 'social', 'history', 'psychology', 'debate', 'argue', 'rights', 'policy', 'government', 'sociology'],
        'description': 'This domain is for those interested in society, human behavior, and justice. Careers include Lawyer, Psychologist, Social Worker, and Policy Analyst.',
        'courses': [
            {'title': 'Introduction to Psychology by Yale', 'url': 'https://www.coursera.org/learn/introduction-psychology'},
            {'title': 'A Law Student\'s Toolkit by Yale', 'url': 'https://www.coursera.org/learn/law-student'}
        ]
    }
}

# --- Helper Functions ---
def preprocess_text(text: str) -> str:
    """Cleans and preprocesses user input text."""
    if not isinstance(text, str): return ""
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(lemmatized_tokens)

def call_gemini_api(prompt: str) -> str:
    """Calls the Google Gemini API for generative tasks."""
    if not GEMINI_API_KEY:
        return "Generative AI feature is not configured. (API key missing)"
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"Gemini API Error: {e}")
        return f"Error connecting to the Generative AI model: {e}"
    except (KeyError, IndexError) as e:
        print(f"Gemini API Response Error: {response.text}")
        return "Could not parse the response from the Generative AI model."

# --- Rasa Actions ---

class ActionStoreName(Action):
    """Stores the user's name and asks for their interests."""
    def name(self) -> Text:
        return "action_store_name"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        name = tracker.latest_message.get('text')
        dispatcher.utter_message(response="utter_ask_interest", name=name)
        return [SlotSet("name", name)]

class ActionSuggestCareer(Action):
    """Analyzes all user inputs and suggests a career."""
    def name(self) -> Text:
        return "action_suggest_career"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        full_profile_text = " ".join([
            tracker.get_slot("user_interests") or "",
            tracker.get_slot("user_strengths") or "",
            tracker.get_slot("user_subjects") or "",
            preprocess_text(tracker.get_slot("resume_keywords") or "")
        ])
        processed_profile = preprocess_text(full_profile_text)

        if not processed_profile.strip():
            dispatcher.utter_message(text="I need more information to make a recommendation. Could you tell me about your interests?")
            return []

        scores = {domain: 0 for domain in CAREER_DATA.keys()}
        vectorizer = TfidfVectorizer()
        for domain_name, data in CAREER_DATA.items():
            domain_keywords = " ".join(data['keywords'])
            corpus = [processed_profile, domain_keywords]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            scores[domain_name] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

        recommended_career = max(scores, key=scores.get)
        
        career_info = CAREER_DATA[recommended_career]
        response = f"### Based on your profile, I recommend exploring **{recommended_career}**!\n\n"
        response += f"**About this field:**\n{career_info['description']}\n\n"
        response += "**Suggested Online Courses to Explore:**\n"
        for course in career_info['courses']:
            response += f"- [{course['title']}]({course['url']})\n"
        
        dispatcher.utter_message(text=response)
        dispatcher.utter_message(response="utter_now_what")
        
        return [SlotSet("recommended_career", recommended_career)]

class ActionGenerateReport(Action):
    """Generates a PDF report summarizing the user's session."""
    def name(self) -> Text:
        return "action_generate_report"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name") or "User"
        recommended_career = tracker.get_slot("recommended_career")
        
        if not recommended_career:
            dispatcher.utter_message("I need to suggest a career first before I can generate a report.")
            return []

        report_path = f"reports/Career_Report_{name.replace(' ', '_')}_{tracker.sender_id[:8]}.pdf"
        os.makedirs("reports", exist_ok=True)

        c = canvas.Canvas(report_path, pagesize=letter)
        width, height = letter
        styles = getSampleStyleSheet()
        style_body = styles['BodyText']
        style_heading = styles['h2']

        # Content Flowable List
        story = []

        # Title
        story.append(Paragraph("Personalized Career Report", styles['h1']))
        story.append(Spacer(1, 24))
        story.append(Paragraph(f"For: {name}", style_heading))
        story.append(Spacer(1, 24))

        # Recommended Career
        story.append(Paragraph("Recommended Career Path:", style_heading))
        story.append(Paragraph(f"<b>{recommended_career}</b>", style_body))
        story.append(Spacer(1, 12))

        # Description
        career_info = CAREER_DATA.get(recommended_career, {})
        story.append(Paragraph("About this Field:", style_heading))
        story.append(Paragraph(career_info.get('description', 'No details available.'), style_body))
        story.append(Spacer(1, 12))
        
        # User Inputs
        story.append(Paragraph("Your Profile Summary:", style_heading))
        story.append(Paragraph(f"<b>Interests:</b> {tracker.get_slot('user_interests')}", style_body))
        story.append(Paragraph(f"<b>Strengths:</b> {tracker.get_slot('user_strengths')}", style_body))
        story.append(Paragraph(f"<b>Subjects:</b> {tracker.get_slot('user_subjects')}", style_body))
        story.append(Spacer(1, 24))

        # Build the PDF
        doc_template = c.beginText(72, height - 72)
        for item in story:
            item.wrapOn(c, width - 144, height)
            item.drawOn(c, 72, height - 100) # Manual positioning for simplicity here
            # A more robust solution uses platypus PageTemplates
        
        c.save()
        
        dispatcher.utter_message(text=f"I have generated your report! You can download it from the sidebar in the app.",
                                 custom={"report_path": report_path})
        
        return [SlotSet("report_path", report_path)]

class ActionSkillGapAnalysis(Action):
    """Compares user's resume to a typical job description using GenAI."""
    def name(self) -> Text:
        return "action_skill_gap_analysis"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        resume_keywords = tracker.get_slot("resume_keywords")
        recommended_career = tracker.get_slot("recommended_career")

        if not resume_keywords or not recommended_career:
            dispatcher.utter_message("I need your resume and a career recommendation first. Please upload your resume from the sidebar.")
            return []

        prompt = f"List the top 15 most important technical skills and soft skills for a '{recommended_career}' role, based on current industry standards. Format as a single comma-separated string. Example: Python, SQL, Communication, Teamwork, Project Management."
        job_skills_str = call_gemini_api(prompt)
        job_skills = set([skill.strip().lower() for skill in job_skills_str.split(',')])
        
        user_skills = set(preprocess_text(resume_keywords).split())
        missing_skills = job_skills - user_skills
        matching_skills = job_skills.intersection(user_skills)

        response = f"### Skill Gap Analysis for a **{recommended_career}** role:\n\n"
        if matching_skills:
            response += f"**✅ Skills you likely have (based on your resume):**\n- " + "\n- ".join(sorted(list(matching_skills))[:7]) + "\n\n"
        if missing_skills:
            response += f"**🎯 Key skills to develop for this role:**\n- " + "\n- ".join(sorted(list(missing_skills))[:7]) + "\n\n"
        response += "Focusing on these areas will significantly strengthen your profile for this career path!"
        
        dispatcher.utter_message(text=response)
        return []

class ActionGenerateDayInLife(Action):
    """Uses GenAI to create a narrative of a day in the life."""
    def name(self) -> Text:
        return "action_generate_day_in_life"
        
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        career = tracker.get_slot("recommended_career")
        if not career:
            dispatcher.utter_message("Please let me recommend a career first.")
            return []

        prompt = f"Create an engaging, first-person narrative of a 'day in the life' of a {career}. Make it realistic, covering daily tasks, challenges, and rewarding moments. Write it in about 150 words."
        
        dispatcher.utter_message(text="🎨 Generating a simulation of a day in this career... this might take a moment.")
        day_in_life_text = call_gemini_api(prompt)
        dispatcher.utter_message(text=f"### A Day in the Life of a {career}:\n\n" + day_in_life_text)
        return []

class ActionMockInterview(Action):
    """Starts a mock interview for the recommended career."""
    def name(self) -> Text:
        return "action_mock_interview"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        career = tracker.get_slot("recommended_career")
        if not career:
            dispatcher.utter_message("I need to know which career to interview you for!")
            return []

        prompt = f"Generate one common behavioral or technical interview question for a {career} role."
        question = call_gemini_api(prompt)

        dispatcher.utter_message(f"Great! Let's start a mock interview for a **{career}** role. Here is your first question:\n\n*'{question}'*")
        return [SlotSet("interview_mode", True)]

class ActionRestart(Action):
    """Resets the conversation and all slots."""
    def name(self) -> Text:
        return "action_restart"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [AllSlotsReset()]