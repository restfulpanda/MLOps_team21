import io
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import timm
import pandas as pd
import altair as alt

MODEL_PATH = Path("./model/best_model.pth")
IMAGE_SIZE = 384
NUM_CLASSES = 19
THRESHOLD = 0.5

LABEL_NAMES = [str(i) for i in range(1, NUM_CLASSES + 1)]

@st.cache_resource(show_spinner=False)
def load_model():
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=False,
        num_classes=0
    )
    
    in_features = model.num_features
    head = nn.Sequential(
        nn.Linear(in_features, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    model.classifier = head
    
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def get_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])

def predict(image: Image.Image,
            model: nn.Module,
            device: torch.device,
            transform: T.Compose,
            threshold: float = THRESHOLD):
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_t)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]

    labels = [LABEL_NAMES[i] for i, p in enumerate(probs) if p > threshold]
    return labels, probs

st.set_page_config(page_title="Multi‑label Image Classifier",
                   page_icon="🖼️",
                   layout="centered")

st.title("📷 Multi‑label Image Classifier")
st.write(
    """
    Загрузите изображение (JPEG / PNG) — модель EfficientNet‑B4 предскажет 
    все подходящие категории.
    """
)   

uploaded_file = st.file_uploader("Выберите файл изображения", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    except Exception as e:
        st.error(f"Не удалось открыть файл как изображение: {e}")
        st.stop()

    st.image(img, caption="Загруженное изображение", use_container_width=True)

    model = load_model()
    device = get_device()
    model.to(device)
    transform = get_transform()

    with st.spinner("Предсказание..."):
        labels, probs = predict(img, model, device, transform)

    st.subheader("Результаты")
    if labels:
        st.success(f"Метки: {', '.join(labels)}")
    else:
        st.warning("Ни одна метка не превысила порог.")

    if st.checkbox("Показать вероятности по всем классам"):

        df = pd.DataFrame({
            "Class": LABEL_NAMES,
            "Probability": probs
        })

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Class:N', sort='-x')
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)