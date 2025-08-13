import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import requests
import json

# Firebase DB URL
dburl = "https://car-rental-60102-default-rtdb.firebaseio.com/feedbacks.json"

def get_feedbacks():
    response = requests.get(dburl)
    data = response.json()
    return data

# Define the path to the sentiment_model directory
model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "sentiment-model", "content", "sentiment-model"))

# Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
tokenizer = DistilBertTokenizer.from_pretrained(model_dir, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_label_map():
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    id2label = config.get("id2label", {})
    return {int(k): v for k, v in id2label.items()}

reverse_label_map = get_label_map()
label_name_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def analyze_and_update_feedbacks():
    feedbacks = get_feedbacks()
    if not feedbacks:
        print("No feedbacks found.")
        return

    for key, entry in feedbacks.items():
        feedback_text = entry.get("feedback")
        if feedback_text:
            # Run sentiment analysis
            inputs = tokenizer(feedback_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).item()
            label = reverse_label_map.get(prediction, str(prediction))
            sentiment = label_name_map.get(label, label)

            # Update feedback in Firebase
            patch_url = f"https://car-rental-60102-default-rtdb.firebaseio.com/feedbacks/{key}.json"
            requests.patch(patch_url, json={"sentiment": sentiment})
            print(f"Updated feedback {key} with sentiment: {sentiment}")

    print("All feedbacks analyzed and updated with sentiment.")

if __name__ == "__main__":
    analyze_and_update_feedbacks()