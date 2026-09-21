# AI-Based Traffic Violation Detector with NLP-Based E-Challan Generation

An AI-based traffic violation detection system combining **Computer Vision, OCR, NLP, SQLite, and PDF generation** to automate the detection and processing of two-wheeler traffic violations.

The system is designed as a **TY BSc AI academic project** and provides a complete pipeline from image-based violation detection to automatic e-challan generation.

---

## 🚀 Live Demo

**Live Streamlit App:**  
https://ai-traffic-violation-detector-9uqxhrbjy5sdsno3gjqi4q.streamlit.app/

**GitHub Repository:**  
https://github.com/barotparth9799-dot/ai-traffic-violation-detector

---

## ✨ Features

- No Helmet Detection
- Triple Riding Detection
- Motorcycle and Person Detection using YOLOv8
- Helmet / No-Helmet Detection
- License Plate Recognition using EasyOCR
- SQLite-based Vehicle and Owner Registry
- NLP-based Violation Message Generation
- spaCy-based NLP Analysis
- Automatic PDF E-Challan Generation
- Evidence Image Capture
- Streamlit Web Interface
- Violation History and Database Records

---

## 🚦 Supported Traffic Violations

The final project detects two traffic violations:

### 1. Riding Without Helmet

Detects riders who are not wearing a helmet.

### 2. Triple Riding

Detects three riders associated with the same motorcycle.

> **Red Light Detection is not part of the final project scope.**

---

## 🧠 System Architecture

```text
                         INPUT IMAGE
                              |
                              v
                    +-------------------+
                    |      YOLOv8       |
                    | Object Detection  |
                    +-------------------+
                       /       |       \
                      /        |        \
                     v         v         v
                 Persons   Motorcycles  Helmet
                                      Detection
                       \        |        /
                        \       |       /
                         v      v      v
                       Violation Detection
                              |
                +-------------+-------------+
                |                           |
                v                           v
          No Helmet                  Triple Riding
                |                           |
                +-------------+-------------+
                              |
                              v
                       Evidence Image
                              |
                              v
                    License Plate OCR
                         EasyOCR
                              |
                              v
                    Vehicle Registry
                         SQLite
                              |
                              v
                      Owner Details
                              |
                              v
                    NLP Message Generation
                          spaCy NLP
                              |
                              v
                    PDF E-Challan Generation
                         ReportLab
                              |
                              v
                     Violation History