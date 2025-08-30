import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import time

# ==========================
# OpenAI API Key
# ==========================
openai.api_key = " "

# ==========================
# Load dataset and model
# ==========================
data = pd.read_csv("Health_dataset.csv")  # Ensure columns: symptom, possible_condition, recommended_action
model = SentenceTransformer('all-MiniLM-L6-v2')
symptom_embeddings = model.encode(data['symptom'].astype(str).tolist())

# ==========================
# Chatbot functions
# ==========================
def health_chatbot_rule(user_input):
    for idx, row in data.iterrows():
        if str(row['symptom']).lower() in user_input.lower():
            return f"**Symptom:** {row['symptom']}\n\n**Possible Condition:** {row['possible_condition']}\n\n**Recommended Action:** {row['recommended_action']}"
    return None

def health_chatbot_gpt(user_input):
    messages = [
        {"role": "system", "content": "You are a helpful health assistant."},
        {"role": "user", "content": f"A user reports the symptom: '{user_input}'. Provide concise, safe, and informative advice. Suggest next steps but do NOT provide a diagnosis."}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def health_chatbot(user_input):
    rule_response = health_chatbot_rule(user_input)
    if rule_response:
        return rule_response
    return health_chatbot_gpt(user_input)

# ==========================
# Streamlit UI Styling
# ==========================
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #DFF6E4;  /* Light green */
}
.chat-bubble {
    background-color:#86C58B;  /* Medium green bot messages */
    padding:15px;
    border-radius:20px;
    max-width:70%;
    margin-bottom:15px;
    box-shadow: 3px 3px 10px rgba(0,0,0,0.2);
    color: #0B3D0B;
    font-size:16px;
}
.user-bubble {
    background-color:#B2F2BB;  /* Light green user messages */
    padding:15px;
    border-radius:20px;
    max-width:70%;
    margin-bottom:15px;
    margin-left:auto;
    box-shadow: 3px 3px 10px rgba(0,0,0,0.15);
    color: #0B3D0B;
    font-size:16px;
}
button.css-1emrehy.edgvbvh3 { 
    background-color: #3AA873 !important; 
    color: #ffffff !important;
    font-weight: bold;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ==========================
# Title and description
# ==========================
st.markdown("<h1 style='text-align:center; color: #0B3D0B;'>🩺 AI-Powered Health Assistant</h1>", unsafe_allow_html=True)
st.write("Ask me about your symptoms and get **informational health advice only**.")

# ==========================
# Sidebar with tabs
# ==========================
st.sidebar.title("ℹ️ Information Center")
info_tab = st.sidebar.radio("Choose a section:", ["None", "About the App", "About the Developer"])

if info_tab == "About the App":
    st.sidebar.markdown("""
    ### 🩺 About the App
    This **AI Health Assistant** helps you:
    - Enter symptoms or select common ones
    - Get possible causes and safe advice
    - Learn recommended next steps  

    ⚠️ **Disclaimer:** This app is for informational purposes only and should **not** replace professional medical advice.
    """)

elif info_tab == "About the Developer":
    st.sidebar.markdown("""
    ### 👨‍💻 About the Developer  
    **Sherriff Abdul-Hamid**  
    AI Engineer | Data Scientist/Analyst | Economist   

    **Contact:**  
    [GitHub](https://github.com/S-ABDUL-AI) |  
    [LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) |  
    📧 Sherriffhamid001@gmail.com
    """)

# ==========================
# Chat Interface
# ==========================
st.subheader("Quick Access: Common Symptoms")
common_symptoms = ["Fever", "Cough", "Headache", "Overweight", "Fatigue", "Stomach Pain"]
cols = st.columns(3)
for i, symptom in enumerate(common_symptoms):
    if cols[i % 3].button(symptom):
        user_input = symptom
        placeholder = st.empty()
        for j in range(3):
            placeholder.markdown(f"⏳ Bot is typing{'.' * (j + 1)}")
            time.sleep(0.5)
        bot_msg = health_chatbot(user_input)
        placeholder.markdown(f"<div class='chat-bubble'>🩺 <b>Bot:</b> {bot_msg}</div>", unsafe_allow_html=True)

# Text input for custom symptom
user_input = st.text_input("💬 Or describe your symptom:")
if user_input:
    st.markdown(f"<div class='user-bubble'>👤 <b>You:</b> {user_input}</div>", unsafe_allow_html=True)
    placeholder = st.empty()
    for i in range(3):
        placeholder.markdown(f"⏳ Bot is typing{'.' * (i+1)}")
        time.sleep(0.5)
    bot_msg = health_chatbot(user_input)
    placeholder.markdown(f"<div class='chat-bubble'>🩺 <b>Bot:</b> {bot_msg}</div>", unsafe_allow_html=True)
