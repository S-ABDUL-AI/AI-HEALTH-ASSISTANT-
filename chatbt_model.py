import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load dataset
data = pd.read_csv("Health_dataset.csv")

# 2. Load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Encode all symptoms in the dataset
symptom_embeddings = model.encode(data['symptom'].astype(str).tolist())


# 4. Chatbot function
def health_chatbot_semantic(user_input):
    input_emb = model.encode([user_input])
    similarities = cosine_similarity(input_emb, symptom_embeddings)[0]
    best_idx = similarities.argmax()

    if similarities[best_idx] > 0.6:  # Confidence threshold
        row = data.iloc[best_idx]
        return {
            "Symptom": row['symptom'],
            "Possible Condition": row['possible_condition'],
            "Recommended Action": row['recommended_action']
        }
    return {"Message": "I'm not sure. Please consult a healthcare professional."}


# 5. Example test
print(health_chatbot_semantic("I feel very tired and have no energy"))
print(health_chatbot_semantic("I am overweight"))
