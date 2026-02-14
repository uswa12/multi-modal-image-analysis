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

```
group-2/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── yolov8n.pt            # YOLOv8 Nano model weights
├── ReadMe.md             # Project documentation
├── venv/                 # Virtual environment directory
├── .streamlit/           # Streamlit configuration
├── .gitignore            # Git ignore file
├── .git/                 # Git repository
└── __pycache__/          # Python cache directory
```


---

## ▶ How to Run Locally

1. **Create a virtual environment:**
```bash
python3 -m venv venv
```

2. **Activate the virtual environment:**
```bash
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open in your browser:**
```
http://localhost:8501
```

6. **Deactivate the virtual environment (when done):**
```bash
deactivate
```

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

