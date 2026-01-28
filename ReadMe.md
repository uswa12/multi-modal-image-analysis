# 🧠 Multimodal Image AI

Multimodal Image AI is a Streamlit-based web application that analyzes images using multiple AI models. Users can upload an image or capture one from a camera to generate a caption, detect objects, recognize emotions, summarize the scene, and create a creative story from the image.

The project integrates Computer Vision and Natural Language Processing into a single multimodal pipeline.

---

## 🚀 Features

- 📸 Upload image or capture from camera  
- 📝 Automatic image caption generation (BLIP)  
- 🎯 Object detection using YOLOv8 Nano  
- 😊 Emotion detection using DeepFace  
- 📊 Scene summary generation  
- 📖 Creative story generation  
- 🎨 Modern UI with custom CSS styling  

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- BLIP (Bootstrapped Language-Image Pretraining)  
- YOLOv8 Nano (Ultralytics)  
- DeepFace  
- OpenCV  
- NumPy  
- Pillow  
- CSS (for UI styling)  

---

## 📂 Project Structure

project/
│
├── app.py
├── requirements.txt
└── README.md


---

## ▶ How to Run Locally

Since no virtual environment is used, install dependencies globally:

```bash
pip install -r requirements.txt
Then run the app:

streamlit run app.py
Open in browser:

http://localhost:8501
🌐 Deployment
This project can be deployed using Streamlit Community Cloud:

Push the project to GitHub

Go to https://share.streamlit.io

Click New App

Select your repository

Choose app.py

Click Deploy

🧠 System Workflow
User Image
   ↓
BLIP → Caption
   ↓
YOLO → Objects
   ↓
DeepFace → Emotion
   ↓
Summary + Story
   ↓
Streamlit UI
⚠ Limitations
Emotion detection requires a visible face.

Performance depends on CPU resources.

Large images may slow processing.

Object detection accuracy varies with lighting and angle.

🔮 Future Improvements
Add video stream processing

Draw bounding boxes on detected objects

Store results in database

Multi-language captioning

GPU acceleration

🎓 Academic Use
This project demonstrates multimodal AI by combining computer vision and NLP for educational and demo purposes.