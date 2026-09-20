# AI-Based Automatic Traffic Violation Detector with NLP-Based E-Challan Generation

An advanced, beginner-friendly **Traffic Safety Enforcement System** combining **Computer Vision (CV)** and **Natural Language Processing (NLP)**. The project is designed specifically for **TY BSc AI / BSc Computer Science** academic submissions and presentations.

It features a high-performance detection pipeline, license plate recognition, an SQLite registration system, dynamic NLP-based legal message parsing, and automatic PDF challan generation—all wrapped inside a premium, dark-themed Streamlit dashboard.

---

## 📁 Folder Structure

```
traffic_violation_system/
├── app.py                   # Main Streamlit web dashboard & UI
├── detector.py              # YOLOv8 object detection (Person, Motorcycle, Helmet)
├── ocr_module.py            # EasyOCR License Plate recognition & preprocessing
├── database.py              # SQLite database manager for owner registry & logs
├── nlp_engine.py            # spaCy NLP text parsing, NER, and POS tagger
├── challan_generator.py     # ReportLab PDF compiler for generating official challans
├── requirements.txt         # Project dependencies
├── README.md                # Setup instructions & presentation notes (this file)
├── data/                    # Contains SQLite database file (traffic_system.db)
├── evidence/                # Captured screenshot image crops of violations
├── challans/                # Compiled PDF E-Challans ready for download
└── sample_media/            # Pre-configured demonstration media
```

---

## ⚙️ Setup and Installation

Follow these steps to run the project on your Windows machine:

### 1. Install Python
Make sure you have **Python 3.8 to 3.11** installed on your system. 

### 2. Open Terminal & Navigate to Project
Open PowerShell or Command Prompt, and navigate to the project directory:
```powershell
cd "C:\Users\parth barot\.gemini\antigravity\scratch\traffic_violation_system"
```

### 3. Create a Virtual Environment (Recommended)
This isolates the project dependencies:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
Install all required libraries specified in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 5. Launch the Dashboard
Run the Streamlit app to start the server:
```powershell
streamlit run app.py
```
This will automatically open the dashboard in your web browser (usually at `http://localhost:8501`).

---

## 🎯 How to Demonstrate the Project in Your Viva

1. **Verify Weights**: When you click the button for the first time, the program will:
   - Load the COCO model `yolov8n.pt` (automatically downloaded by Ultralytics).
   - Attempt to download `iam-tsr/yolov8n-helmet-detection/best.pt` from Hugging Face for helmet vs. no-helmet.
   - *Note:* If you are offline, the system will seamlessly run in **Simulation Fallback Mode** so the dashboard still behaves perfectly for your examiners!
2. **Execute Detection**: Go to the **Traffic Violation Detection** tab and click **Run AI Pipeline** on the preloaded sample image.
3. **Download PDF**: Review the detected boxes, the OCR license plate parsing, the owner details fetched from SQLite, and click the **Download PDF E-Challan** button to view the compiled document.
4. **Linguistic Deep-dive**: Switch to the **NLP & Linguistic Analysis** tab. Show how spaCy tokenizes the text and performs Named Entity Recognition (NER) to verify names, plates, and fines.
5. **Database Ledger**: Switch to the **SQLite Database Registry** tab. Show the current vehicle owners list and history logs. You can register new vehicles via the sidebar to see OCR auto-correction matching details in real time!

---

## 🎓 Academic Concepts & Project Explanation (For presentation)

### A. Computer Vision Pipeline (CV)
1. **Object Detection**:
   - **YOLOv8** (You Only Look Once) is chosen due to its state-of-the-art inference speed and accuracy on edge devices.
   - We utilize two model passes: one for global context (finding motorcycles and persons) and one for class safety categorization (finding helmets or no-helmets).
2. **Rider Bounding Box Overlap Logic**:
   - A motorcyclist is identified by measuring the overlap between a detected `person` box and a `motorcycle` box. If the person's bounding box intersects with the motorcycle's box by more than 15%, the person is classified as a rider.
   - If a `no-helmet` box is detected within the rider's upper boundaries (head area), a safety violation is flagged.
3. **License Plate Recognition (LPR)**:
   - The violating vehicle area is cropped. The lower portion is preprocessed using **OpenCV** (Grayscaling, Bilateral Noise Filter, and Bicubic Interpolation resizing) to maximize text contrast.
   - **EasyOCR** (a deep learning OCR based on ResNet and LSTM) reads characters.
   - **Regular Expressions (Regex)** clean and filter the OCR string to verify standard Indian vehicle formats: `[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}` (e.g. `MH12AB1234`).

### B. Natural Language Processing Module (NLP)
1. **Dynamic Generation**:
   - Instead of static text templates, raw variables (Owner, Plate Number, Offence, Fine, Date) are parsed into a synthetically framed message.
2. **Linguistic Parsing**:
   - The notice text is passed through **spaCy's `en_core_web_sm` pipeline** to show the NLP engine's understanding.
   - **Tokenization**: Splitting sentences into grammatically meaningful words.
   - **POS (Parts-of-Speech) Tagging**: Highlighting Nouns (`PROPN`), Verbs (`VERB`), and Adjectives. This proves structural syntax check.
   - **NER (Named Entity Recognition)**: Isolates the offender (`PERSON`), the fine amount (`MONEY`), and the plate number (`VEHICLE_NO`).

### C. System Architecture & Output compilation
1. **Database Registry**:
   - Powered by **SQLite**, which is lightweight and local.
   - Registry tables match number plates with owner name, contact, and address.
2. **Document Compiler**:
   - Powered by **ReportLab Platypus**.
   - Automatically arranges margins, inserts text tables, renders the NLP notice, and embeds the cropped evidence image of the violation directly into a downloadable PDF document.
