import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import numpy as np
import cv2
from deepface import DeepFace
import random

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Multimodal Image AI", layout="wide")

# ---------------- STYLING ---------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

body {
    background: linear-gradient(135deg, #020617, #0f172a, #1e293b);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

/* Cards */
.card {
    background: linear-gradient(145deg, #0f172a, #1e293b);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,.5);
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,.08);
}

/* Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    background: linear-gradient(to right, #38bdf8, #a78bfa, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

/* Tags */
.tag {
    display: inline-block;
    background: linear-gradient(to right, #22c55e, #38bdf8);
    color: black;
    padding: 6px 14px;
    border-radius: 20px;
    margin: 4px;
    font-size: 14px;
    font-weight: 600;
}

/* Story Box */
.story-box {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,.4);
}

/* Emotion pill */
.emotion {
    background: linear-gradient(to right, #facc15, #22c55e);
    padding: 8px 18px;
    border-radius: 25px;
    display: inline-block;
    color: black;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(to right, #6366f1, #22c55e);
    color: white;
    border-radius: 14px;
    padding: 0.6em 1.4em;
    font-size: 16px;
    border: none;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(99,102,241,.7);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #020617;
    border-radius: 14px;
    padding: 10px;
}

/* Image glow */
img {
    border-radius: 16px;
    box-shadow: 0 0 25px rgba(56,189,248,.35);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ---------------- #
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    yolo = YOLO("yolov8n.pt")
    return processor, blip_model, yolo

processor, blip_model, yolo = load_models()

# ---------------- FUNCTIONS ---------------- #
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)

def detect_objects(image, caption=""):
    results = yolo(image, conf=0.5)
    if not results[0].boxes:
        return []

    names = results[0].names
    boxes = results[0].boxes
    confidences = boxes.conf.tolist()
    classes = boxes.cls.tolist()

    detections = [(names[int(cls_id)], conf) for cls_id, conf in zip(classes, confidences)]
    high_conf_objects = [obj for obj, conf in detections if conf > 0.55]

    seen = set()
    objects = []
    for obj in high_conf_objects:
        if obj not in seen:
            objects.append(obj)
            seen.add(obj)

    false_positive_filters = {
        'paper': ['tv', 'monitor', 'laptop', 'keyboard', 'mouse'],
        'book': ['tv', 'monitor', 'screen'],
        'desk': ['tv'],
        'person': [],
        'cup': ['bowl', 'plate'],
    }

    caption_lower = caption.lower()
    for keyword, false_positives in false_positive_filters.items():
        if keyword in caption_lower:
            objects = [obj for obj in objects if obj not in false_positives]

    return objects if objects else ["scene"]

def generate_summary(caption, objects):
    return f"This image shows {caption}. Key visible elements include {', '.join(objects[:5])}."

def detect_emotion_from_image(image, objects):
    if "person" not in objects:
        return "N/A (No person detected)"

    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = DeepFace.analyze(
            img_path=img,
            actions=["emotion"],
            enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]
        emotion = result["dominant_emotion"]
        emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😮',
            'disgust': '🤢',
            'neutral': '😐'
        }
        emoji = emotion_emojis.get(emotion.lower(), '😐')
        return emotion.capitalize() + " " + emoji
    except:
        return "Emotion detection failed"

def generate_story(caption, objects):
    stories = [
        f"""
Once upon a moment, {caption}. Around the scene were {', '.join(objects[:4])}, quietly shaping the atmosphere.
The air felt alive, as if something important was about to happen. People moved with purpose, unaware they were part
of a larger story. That single frame captured not just an image, but a memory in the making.
""",
        f"""
In that instant captured in time, {caption}. The presence of {', '.join(objects[:3])} told a deeper narrative.
Every element had arrived exactly when it needed to. This wasn't chance—it was a carefully orchestrated moment where
everything came together in perfect harmony. A story written not in words, but in presence and arrangement.
""",
        f"""
A story unfolded in a single frame: {caption}. Amidst the {', '.join(objects[:4])}, there was meaning waiting to be discovered.
Life doesn't always announce its significance. Sometimes it whispers through ordinary moments, through objects placed just so,
through the silent presence of those who inhabit these spaces. This image was one such whisper.
""",
        f"""
Consider this scene: {caption}. The details matter—{', '.join(objects[:3])} each contributed to a larger truth.
Without fanfare or explanation, the image speaks to the attentive observer. It's a reminder that profound stories
hide in plain sight, waiting for someone to pause and truly see what's there.
""",
        f"""
Within this frame lies a story: {caption}. Surrounded by {', '.join(objects[:4])}, the moment became eternal.
These weren't just random objects and faces—they were characters in an unfolding narrative. Each element played its part,
and together they created something that transcended the mere act of photography.
"""
    ]
    return random.choice(stories)

# ---------------- UI ---------------- #
st.markdown("<div class='title'>🧠 Multimodal Image AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image and let AI understand, feel, and tell its story</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_container_width=True)
        analyze_btn = st.button("🚀 Analyze Image", use_container_width=True)
    else:
        analyze_btn = False
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if analyze_btn:
        with st.spinner("Analyzing with AI..."):
            caption = generate_caption(image)
            objects = detect_objects(image, caption)
            summary = generate_summary(caption, objects)
            emotion = detect_emotion_from_image(image, objects)
            story = generate_story(caption, objects)

            st.session_state.caption = caption
            st.session_state.objects = objects
            st.session_state.summary = summary
            st.session_state.emotion = emotion
            st.session_state.story = story

if "caption" in st.session_state:
    st.markdown("<hr style='border:1px solid #38bdf8'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='card'><h4>📝 Caption</h4>", unsafe_allow_html=True)
        st.write(st.session_state.caption)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'><h4>📊 Summary</h4>", unsafe_allow_html=True)
        st.write(st.session_state.summary)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='card'><h4>😊 Emotion</h4>", unsafe_allow_html=True)
        st.markdown(f"<div class='emotion'>{st.session_state.emotion}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h4>🏷 Detected Objects</h4>", unsafe_allow_html=True)
    for obj in st.session_state.objects:
        st.markdown(f"<span class='tag'>{obj}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='story-box'><h4>📖 Story</h4>", unsafe_allow_html=True)
    st.write(st.session_state.story)
    if st.button("✨ Regenerate Story", use_container_width=True):
        st.session_state.story = generate_story(st.session_state.caption, st.session_state.objects)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
