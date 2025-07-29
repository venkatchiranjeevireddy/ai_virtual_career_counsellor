# 🧠 AI Virtual Career Counsellor

An NLP-based chatbot that recommends career paths based on user interests using Rasa and Streamlit.

---

## 🎯 Objective

Build an interactive virtual counsellor that suggests career options (e.g., engineering, arts, commerce, etc.) based on a user's responses.

---

## 🛠️ Tools & Technologies

- **Python**
- **NLTK** – Text preprocessing  
- **Rasa** – NLP chatbot development  
- **Streamlit** – Frontend interface  
- **Streamlit Cloud** – Deployment

---

## 📁 Project Structure# Project documentation
---

## 🚀 How It Works

1. **Define Intents:**  
   Create intent examples like:
   - `interest_tech`
   - `interest_arts`
   - `interest_commerce`

2. **Preprocess Input (NLTK):**  
   - Tokenize, lemmatize, and clean user inputs before feeding into Rasa.

3. **Rasa Training:**  
   - Train using `rasa train`.
   - Dialogue rules guide how bot responds to different intents.

4. **Career Logic (actions.py):**  
   - Map intents to career suggestions.
   - Example: `"interest_tech"` → Suggest: Software Engineer, Data Scientist

5. **Streamlit Frontend:**  
   - Build chat interface with `streamlit_app.py`
   - Send user messages and show bot responses.

6. **Deployment:**  
   - Push code to GitHub.
   - Deploy Streamlit frontend using **Streamlit Cloud**.

---

## 🧪 Sample Use Case

**User Input:**  
"I love building apps and solving technical problems"

**Bot Response:**  
"You might enjoy careers like Software Engineer, Data Scientist, or AI Researcher."

---

## 📦 Deliverables

- ✅ Rasa Project Folder (`data/`, `domain.yml`, `actions/`, etc.)  
- ✅ Streamlit Frontend (`streamlit_app.py`)  
- ✅ Sample Interaction Video (Demo of working chatbot)

---

## 💡 Future Improvements

- Add more intents (medicine, law, business, etc.)  
- Integrate personality/aptitude quiz  
- Store responses for analysis  
- Add user authentication (basic login system)

---


## 🧑‍💻 Author
venkat chiranjeevi reddy 
