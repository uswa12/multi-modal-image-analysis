import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import numpy as np
import cv2
from deepface import DeepFace
import random

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Multimodal Image AI", layout="wide")

# ---------------- STYLING (FIXED VISIBILITY) ---------------- #
st.markdown("""
<style>
html, body {
    background-color: #0f172a;
    color: white;
}

/* Card */
.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
}

/* Title */
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

/* Headings */
.card h4 {
    color: #22c55e !important;
}

/* Tags */
.tag {
    display: inline-block;
    background: #22c55e;
    color: black;
    padding: 6px 14px;
    border-radius: 20px;
    margin: 5px;
    font-size: 14px;
    font-weight: bold;
}

/* Emotion pill */
.emotion {
    background: #facc15;
    color: black;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}

/* Story box */
.story-box {
    background: #1e293b;
    padding: 30px;
    border-radius: 15px;
    margin-top: 30px;
}

.story-box h4 {
    color: #38bdf8 !important;
    text-align: center;
}

.story-text {
    color: black !important;
    font-size: 18px;
    line-height: 1.8;
    text-align: justify;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ---------------- #
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    yolo = YOLO("yolov8n.pt")
    return processor, model, yolo

processor, blip_model, yolo = load_models()

# ---------------- FUNCTIONS ---------------- #
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)

def detect_objects(image):
    results = yolo(image, conf=0.5)
    if not results[0].boxes:
        return ["scene"]

    names = results[0].names
    boxes = results[0].boxes
    classes = boxes.cls.tolist()
    confidences = boxes.conf.tolist()

    objects = list(set(
        names[int(cls_id)] for cls_id, conf in zip(classes, confidences) if conf > 0.55
    ))

    return objects if objects else ["scene"]

def detect_emotion(image, objects):
    if "person" not in objects:
        return "N/A (No person)"

    try:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        return result["dominant_emotion"].capitalize()
    except:
        return "Emotion detection failed"

def generate_story(caption, objects):
    obj_text = ", ".join(objects[:4])

    stories = [

        # Story 1 – Cinematic
        f"""
The moment feels almost cinematic. {caption.capitalize()}.
With {obj_text} shaping the surroundings, the atmosphere carries a quiet intensity.
There’s something unspoken here — a pause in time where emotions,
movement, and meaning converge into a powerful visual narrative.
        """,

        # Story 2 – Emotional & Reflective
        f"""
At first glance, it may seem simple: {caption}.
But the presence of {obj_text} adds depth to the scene.
It feels like a fleeting memory captured mid-breath —
a reminder that even ordinary moments can hold extraordinary emotion.
        """,

        # Story 3 – Descriptive & Immersive
        f"""
In this scene, {caption}.
Around it, {obj_text} define the space and give it structure.
The light, expressions, and subtle details combine
to create a vivid snapshot of life unfolding naturally and authentically.
        """,

        # Story 4 – Dramatic Tone
        f"""
There’s a quiet drama embedded in this image. {caption.capitalize()}.
Surrounded by {obj_text}, the composition feels deliberate and alive.
It’s as though the world paused for a second,
allowing this exact combination of elements to tell its own story.
        """,

        # Story 5 – Poetic Style
        f"""
A single frame, yet it speaks volumes. {caption.capitalize()}.
Between {obj_text}, meaning quietly emerges.
Shadows and light blend seamlessly,
turning this captured instant into something almost poetic.
        """
    ]

    return random.choice(stories)


# ---------------- HEADER ---------------- #
st.markdown("<div class='title'>🧠 Multimodal Image AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload or capture an image and let AI analyze it intelligently</div>", unsafe_allow_html=True)

# ---------------- TABS ---------------- #
tab1, tab2 = st.tabs(["📤 Upload Image", "📹 Live Camera"])

# ---------------- UPLOAD TAB ---------------- #
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h4>📸 Upload Image</h4>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, width="stretch")

            if st.button("🚀 Analyze Uploaded Image", width="stretch"):
                with st.spinner("Analyzing..."):
                    caption = generate_caption(image)
                    objects = detect_objects(image)
                    emotion = detect_emotion(image, objects)
                    story = generate_story(caption, objects)

                    st.session_state.update({
                        "image": image,
                        "caption": caption,
                        "objects": objects,
                        "emotion": emotion,
                        "story": story
                    })
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
        <h4>ℹ️ How It Works</h4>
        <p style='color:white; font-size:16px;'>
        1️⃣ BLIP generates a smart caption.<br><br>
        2️⃣ YOLO detects visible objects.<br><br>
        3️⃣ DeepFace analyzes facial emotion.<br><br>
        4️⃣ A creative story is generated from the scene.
        </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- LIVE CAMERA TAB ---------------- #
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h4>📹 Capture From Camera</h4>", unsafe_allow_html=True)

        camera_image = st.camera_input("Take a picture")

        if camera_image:
            image = Image.open(camera_image).convert("RGB")
            st.image(image, width=400)

            if st.button("🚀 Analyze Camera Image", width="stretch"):
                with st.spinner("Analyzing..."):
                    caption = generate_caption(image)
                    objects = detect_objects(image)
                    emotion = detect_emotion(image, objects)
                    story = generate_story(caption, objects)

                    st.session_state.update({
                        "image": image,
                        "caption": caption,
                        "objects": objects,
                        "emotion": emotion,
                        "story": story
                    })
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
        <h4>ℹ️ How It Works (Live Camera)</h4>
        <p style='color:white; font-size:16px;'>
        1️⃣ Your browser opens the live camera.<br><br>
        2️⃣ You capture a real-time image.<br><br>
        3️⃣ The captured frame is sent to the AI models.<br><br>
        4️⃣ BLIP generates a caption.<br><br>
        5️⃣ YOLO detects objects in the scene.<br><br>
        6️⃣ If a person is detected, DeepFace analyzes emotion.<br><br>
        7️⃣ A story is generated from the live capture.
        </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- RESULTS ---------------- #
if "caption" in st.session_state:

    st.markdown("---")
    st.image(st.session_state.image, width="stretch")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='card'><h4>Caption</h4>", unsafe_allow_html=True)
        st.write(st.session_state.caption)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'><h4>Detected Objects</h4>", unsafe_allow_html=True)
        for obj in st.session_state.objects:
            st.markdown(f"<span class='tag'>{obj}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='card'><h4>Emotion</h4>", unsafe_allow_html=True)
        st.markdown(f"<div class='emotion'>{st.session_state.emotion}</div>",
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='story-box'><h4>📖 Your Story</h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='story-text'>{st.session_state.story}</div>",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("✨ Regenerate Story", width="stretch"):
        st.session_state.story = generate_story(
            st.session_state.caption,
            st.session_state.objects
        )
        st.rerun()
