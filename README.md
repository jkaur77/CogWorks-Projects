# CogWorks Projects 🧠

A collection of AI and Machine Learning projects built during MIT Beaver Works/

## 📁 Repository Structure
```
CogWorks-Projects/
├── wk1_cp/          # Week 1 — Song Recognition Library
├── wk2_cp/          # Week 2 — Facial Recognition
├── wk3_cp/          # Week 3 — Semantic Image Search
├── capstone/        # Capstone — Food Detection & Nutrition App
│   ├── Food-Detection/
│   │   ├── YOLO/           # YOLOv7 model weights & config
│   │   └── Detectron-Trash/# Detectron2 experiments
└── ryan-sus/        # Team workspace
```

## 🗂️ Projects

### Week 1 — Song Recognition Library
Built an audio fingerprinting system inspired by Shazam. Implemented spectrogram 
generation, peak detection, and hash-based song matching to identify songs from 
short audio clips.

**Key concepts:** FFT, spectrograms, audio fingerprinting, hash maps

---

### Week 2 — Facial Recognition
Developed a facial recognition pipeline using descriptor-based face embeddings. 
Implemented face detection, alignment, and nearest-neighbor matching for 
identity classification.

**Key concepts:** HOG descriptors, face embeddings, KNN classification, OpenCV

---

### Week 3 — Semantic Image Search
Built a semantic search engine that retrieves images based on natural language 
queries using learned visual-text embeddings.

**Key concepts:** CLIP-style embeddings, cosine similarity, semantic search

---

### Capstone — Food Detection & Nutrition App 🍎
The flagship project of the program. A mobile application that tracks nutritional 
value from scanned food images in real time.

- Trained **YOLOv7** object detection model on a custom dataset of **50,000+ 
food images**, achieving **95% validation accuracy**
- Integrated predictions with a **15,000+ item Food API** to generate real-time 
nutritional breakdowns


## 🚀 How to Run

### Prerequisites
```bash
pip install torch torchvision numpy scipy matplotlib jupyter
```

### Run any weekly project
```bash
cd wk1_cp   # or wk2_cp, wk3_cp
jupyter notebook
```

### Run capstone (YOLOv7 inference)
```bash
cd capstone/Food-Detection/YOLO
python detect.py --weights best.pt --source your_image.jpg
```
