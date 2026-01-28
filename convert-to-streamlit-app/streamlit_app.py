# streamlit_app.py
import json, joblib, numpy as np, torch, torch.nn as nn
import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

st.set_page_config(page_title="20 Newsgroups Classifier", layout="wide")

# ---------------------------------------------------------
# Load model + vectorizer + labels
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    vectorizer = joblib.load("vectorizer.pkl")
    with open("label_names.json") as f:
        label_names = json.load(f)

    class NewsMLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.0),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.0),
                nn.Linear(256, num_classes)
            )
        def forward(self, x): return self.net(x)

    model = NewsMLP(
        input_dim=vectorizer.max_features or len(vectorizer.vocabulary_),
        num_classes=len(label_names)
    )
    model.load_state_dict(torch.load("model_state_dict.pt", map_location="cpu"))
    model.eval()

    return vectorizer, label_names, model

vectorizer, label_names, model = load_resources()

# ---------------------------------------------------------
# Load dataset for browsing
# ---------------------------------------------------------
@st.cache_resource
def load_dataset():
    data = fetch_20newsgroups(subset="test", remove=("headers","footers","quotes"))
    return data.data, data.target, data.target_names

docs, targets, target_names = load_dataset()

# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict(texts):
    X = vectorizer.transform(texts).toarray()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
    return preds, probs

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("20 Newsgroups Text Classifier (PyTorch + TF‑IDF)")
st.caption("Browse real documents or enter your own text.")

# Sidebar: dataset browsing
st.sidebar.header("Dataset Browser")

mode = st.sidebar.radio(
    "Choose document source:",
    ["Enter custom text", "Pick from dataset"]
)

# ---------------------------------------------------------
# MODE 1: USER ENTERS TEXT
# ---------------------------------------------------------
if mode == "Enter custom text":
    with st.form("predict_custom"):
        text = st.text_area("Paste text", height=200)
        submitted = st.form_submit_button("Classify")

    if submitted and text.strip():
        pred, probs = predict([text])
        label = label_names[int(pred[0])]

        st.subheader(f"Prediction: {label}")

        top = np.argsort(-probs[0])[:5]
        df = pd.DataFrame({
            "label": [label_names[i] for i in top],
            "probability": [float(probs[0][i]) for i in top]
        })
        st.bar_chart(df, x="label", y="probability")

# ---------------------------------------------------------
# MODE 2: PICK DOCUMENT FROM DATASET
# ---------------------------------------------------------
else:
    st.sidebar.subheader("Select a document")

    # Choose by index
    doc_index = st.sidebar.number_input(
        "Document index",
        min_value=0,
        max_value=len(docs)-1,
        value=0,
        step=1
    )

    # Show the document
    st.subheader(f"Document #{doc_index}")
    st.write(f"**True category:** {target_names[targets[doc_index]]}")
    st.write("---")
    st.write(docs[doc_index])

    # Classify button
    if st.button("Classify this document"):
        pred, probs = predict([docs[doc_index]])
        label = label_names[int(pred[0])]

        st.subheader(f"Predicted: {label}")

        top = np.argsort(-probs[0])[:5]
        df = pd.DataFrame({
            "label": [label_names[i] for i in top],
            "probability": [float(probs[0][i]) for i in top]
        })
        st.bar_chart(df, x="label", y="probability")