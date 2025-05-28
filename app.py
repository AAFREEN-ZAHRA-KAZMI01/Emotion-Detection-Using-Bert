import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# --- Setup ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion columns
emotion_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

idx2emotion = {i: e for i, e in enumerate(emotion_columns)}
emotion2emoji = {
    'admiration': "👏", 'amusement': "😂", 'anger': "😡", 'annoyance': "😒", 'approval': "👍",
    'caring': "🤗", 'confusion': "😕", 'curiosity': "🤔", 'desire': "😍", 'disappointment': "😞",
    'disapproval': "👎", 'disgust': "🤢", 'embarrassment': "😳", 'excitement': "🤩", 'fear': "😱",
    'gratitude': "🙏", 'grief': "😭", 'joy': "😃", 'love': "❤️", 'nervousness': "😬", 'optimism': "😊",
    'pride': "😌", 'realization': "💡", 'relief': "😌", 'remorse': "😔", 'sadness': "😢",
    'surprise': "😮", 'neutral': "😐"
}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class MultiLabelBERT(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

model = MultiLabelBERT(len(emotion_columns))
model.load_state_dict(torch.load("model_trained.pth", map_location=device))
model.to(device)
model.eval()

# --- Streamlit UI ---

st.title("🎭 Multi-Label Emotion Recognition")
st.write("Enter your text below and discover the emotions present!")

text_input = st.text_area("Your Text", height=100)

threshold = st.slider("Prediction Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

if st.button("Predict Emotions"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        tokens = tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64
        )
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        preds = [idx2emotion[i] for i, p in enumerate(probs) if p > threshold]

        if preds:
            st.markdown("### Predicted Emotions:")
            for emotion in preds:
                emoji = emotion2emoji.get(emotion, "")
                st.write(f"{emoji} **{emotion.capitalize()}**")
        else:
            st.info("😐 Neutral (No strong emotions detected above the threshold)")
