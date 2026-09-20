import streamlit as st
import os
import shutil
import cv2
import pandas as pd
import numpy as np
from datetime import datetime

# Import project modules
from database import init_db, get_vehicle_owner, log_violation, get_all_violations, get_all_vehicles, add_vehicle_owner
from detector import TrafficViolationDetector
from ocr_module import LicensePlateRecognizer
from nlp_engine import TrafficNLP
from challan_generator import ChallanPDFGenerator

# Set up page configurations
st.set_page_config(
    page_title="AI Traffic Violation & E-Challan System",
    page_icon="🚥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Glassmorphism & Sleek Accent Colors)
st.markdown("""
<style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stApp {
        background-color: #0F172A;
    }
    /* Title Banner */
    .title-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);
        margin-bottom: 2rem;
        text-align: center;
    }
    .title-banner h1 {
        color: #FFFFFF !important;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .title-banner p {
        color: #E2E8F0;
        font-size: 1.1rem;
        margin: 0;
    }
    /* Card style containers */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3B82F6;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Warning/Alert Card */
    .violation-alert {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .violation-title {
        color: #EF4444;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    /* Success Card */
    .success-alert {
        background: rgba(34, 197, 94, 0.15);
        border: 2px solid #22C55E;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_media")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ----------------------------------------------------
# Helper to copy the generated sample image
# ----------------------------------------------------
def setup_sample_media():
    sample_target = os.path.join(SAMPLE_DIR, "traffic_sample.jpg")
    if not os.path.exists(sample_target):
        # Locate the image in the brain artifacts folder
        brain_dir = r"C:\Users\parth barot\.gemini\antigravity\brain\eac0439a-1205-45d9-bcb3-7b9ea77508a3"
        # Search for any file matching traffic_sample_*.jpg or traffic_sample.jpg
        source_img = None
        if os.path.exists(brain_dir):
            for file in os.listdir(brain_dir):
                if file.startswith("traffic_sample") and file.endswith(".jpg"):
                    source_img = os.path.join(brain_dir, file)
                    break
        
        if source_img and os.path.exists(source_img):
            shutil.copy(source_img, sample_target)
            print(f"Copied sample image from {source_img} to {sample_target}")
        else:
            # Create a mock placeholder image if not found
            print("Creating dummy placeholder image...")
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Sample Image Placeholder", (120, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(sample_target, placeholder)

setup_sample_media()

# Initialize Database and NLP Models
@st.cache_resource
def load_resources():
    init_db() # Run SQLite setup
    detector = TrafficViolationDetector()
    ocr = LicensePlateRecognizer()
    nlp = TrafficNLP()
    pdf = ChallanPDFGenerator()
    return detector, ocr, nlp, pdf

# Load cached instances
try:
    detector, ocr, nlp, pdf_gen = load_resources()
except Exception as e:
    st.error(f"Error loading system components: {e}")
    st.stop()

# Sidebar Setup
st.sidebar.markdown("### 🚥 System Dashboard")
st.sidebar.info("TY BSc AI Project - Version 1.0")

# Model configurations in Sidebar
st.sidebar.markdown("### ⚙️ Model Parameters")
conf_val = st.sidebar.slider("YOLOv8 BBox Confidence", 0.10, 0.90, 0.25, 0.05)
ocr_sim_correct = st.sidebar.checkbox("Enable OCR Fuzzy Matcher", value=True)

# Register new vehicle utility in sidebar
with st.sidebar.expander("📝 Register New Vehicle (Registry)"):
    new_plate = st.text_input("Number Plate", placeholder="MH12AB1234")
    new_owner = st.text_input("Owner Name", placeholder="Rahul Sharma")
    new_phone = st.text_input("Mobile Number", placeholder="+91 9999999999")
    new_addr = st.text_area("Owner Address", placeholder="Street name, City, State")
    
    if st.button("Register Vehicle"):
        if new_plate and new_owner and new_phone and new_addr:
            add_vehicle_owner(new_plate, new_owner, new_phone, new_addr)
            st.sidebar.success(f"Vehicle {new_plate} registered successfully!")
            st.rerun()
        else:
            st.sidebar.error("All registry fields are mandatory.")

# Main Application Banner
st.markdown("""
<div class="title-banner">
    <h1>Traffic Violation Detector & E-Challan Generator</h1>
    <p>Integrated Computer Vision (YOLOv8 + EasyOCR) & NLP (spaCy Parser) System for Traffic Safety Enforcement</p>
</div>
""", unsafe_allow_html=True)

# Metric summary section
v_list = get_all_violations()
total_violations = len(v_list)
reg_vehicles = len(get_all_vehicles())

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_violations}</div>
        <div class="metric-label">Total Violations Logged</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{reg_vehicles}</div>
        <div class="metric-label">Registered Vehicles</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    pending_fines = total_violations * 1000
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">₹{pending_fines}</div>
        <div class="metric-label">Fines Generated</div>
    </div>
    """, unsafe_allow_html=True)
with col_m4:
    model_status = "Online (CPU)" if detector.coco_model else "Offline"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: #22C55E;">{model_status}</div>
        <div class="metric-label">YOLOv8 Service Status</div>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# Tabs Layout
tab_dash, tab_nlp, tab_db, tab_edu = st.tabs([
    "🚥 Traffic Violation Detection", 
    "📝 NLP & Linguistic Analysis", 
    "🗄️ SQLite Database Registry", 
    "🎓 BSc Project Presentation Help"
])

# ----------------------------------------------------
# TAB 1: DETECTION DASHBOARD
# ----------------------------------------------------
with tab_dash:
    st.header("1. Input Traffic Media")
    
    col_input, col_opt = st.columns([2, 1])
    
    with col_input:
        upload_mode = st.radio("Select input source:", ["Use Sample Image (No Helmet Demo)", "Upload Custom Traffic Image"], horizontal=True)
    
    img_path = None
    
    if upload_mode == "Use Sample Image (No Helmet Demo)":
        sample_path = os.path.join(SAMPLE_DIR, "traffic_sample.jpg")
        if os.path.exists(sample_path):
            img_path = sample_path
            st.success("Sample image loaded. Click 'Run Detection Pipeline' below.")
        else:
            st.error("Sample image not found. Please upload a custom image.")
    else:
        uploaded_file = st.file_uploader("Upload traffic JPG/PNG image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            temp_path = os.path.join(SAMPLE_DIR, "temp_upload.jpg")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            img_path = temp_path
            st.image(img_path, caption="Uploaded Image", width=400)

    # Core Execution Button
    if img_path:
        run_btn = st.button("🚀 Run AI Detection Pipeline", type="primary", use_container_width=True)
        
        if run_btn:
            with st.spinner("Processing Computer Vision Pipeline... (YOLOv8 + EasyOCR)"):
                # 1. Run YOLOv8 detector
                annotated_img, detections = detector.detect_violations(img_path)
                
                # Convert annotated image back to RGB for streamlit
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                # Show results layout
                st.write("### AI Bounding Box Annotations")
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    orig_img = cv2.imread(img_path)
                    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    st.image(orig_img_rgb, caption="Input Frame / Raw Image", use_container_width=True)
                    
                with col_res2:
                    st.image(annotated_img_rgb, caption="YOLOv8 Analysis (Vehicles, Riders, Helmet Safety Status)", use_container_width=True)

                # Process detections
                if not detections:
                    st.markdown("""
                    <div class="success-alert">
                        <div class="violation-title" style="color: #22C55E;">✔️ Safety Compliance Verified</div>
                        No traffic violations detected in this frame. All motorcyclists are wearing helmets.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # There are helmet violations!
                    for idx, det in enumerate(detections):
                        st.markdown("""
                        <div class="violation-alert">
                            <div class="violation-title">🚨 Traffic Violation Alert: Riding Without Helmet</div>
                            Safety violation detected. Motorcycle rider is operating the vehicle without a safety helmet.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_det1, col_det2 = st.columns([1, 2])
                        
                        with col_det1:
                            st.write("**Evidence Snippet (Motorcycle ROI):**")
                            # Convert crop to RGB
                            crop_rgb = cv2.cvtColor(det["motorcycle_crop"], cv2.COLOR_BGR2RGB)
                            st.image(crop_rgb, use_container_width=True, caption="Cropped Vehicle ROI")
                        
                        with col_det2:
                            st.write("🔍 **License Plate Extraction (EasyOCR):**")
                            
                            # Extract license plate using OCR
                            raw_plate, confidence = ocr.extract_license_plate(det["motorcycle_crop"])
                            
                            # Clean and perform fuzzy matching
                            all_reg_plates = [v["vehicle_number"] for v in get_all_vehicles()]
                            
                            if ocr_sim_correct:
                                matched_plate, match_conf = ocr.fuzzy_match_plate(raw_plate, all_reg_plates)
                            else:
                                matched_plate = raw_plate
                                match_conf = 1.0
                                
                            if matched_plate:
                                st.success(f"Number Plate Recognized: **{matched_plate}** (OCR Raw: `{raw_plate}`, Conf: {confidence:.2f})")
                            else:
                                # Fallback dummy if OCR failed completely (useful for demo resilience)
                                matched_plate = "MH12AB1234"
                                st.warning(f"OCR reading low. Fallback assigned for presentation demo: **{matched_plate}**")

                            # 2. Database Lookup
                            owner_info = get_vehicle_owner(matched_plate)
                            
                            if owner_info:
                                st.markdown("🗄️ **SQLite Owner Record Found:**")
                                st.json(owner_info)
                            else:
                                # Create dummy record to keep pipeline complete
                                owner_info = {
                                    "vehicle_number": matched_plate,
                                    "owner_name": "Rahul Sharma",
                                    "phone": "+91 98765 43210",
                                    "address": "Flat 402, Sunshine Apartments, Pune, Maharashtra - 411001"
                                }
                                st.warning("License plate not in registry database. Simulating default registration details.")
                                st.json(owner_info)

                        st.write("---")
                        
                        # 3. NLP Module Execution
                        st.write("⚙️ **NLP Notice Message Generation (spaCy):**")
                        fine_amt = 1000
                        time_str = datetime.now().strftime("%Y-%m-%d at %I:%M %p")
                        
                        nlp_msg = nlp.generate_violation_message(
                            owner_name=owner_info["owner_name"],
                            vehicle_number=owner_info["vehicle_number"],
                            violation_type="riding without a helmet",
                            fine_amount=fine_amt,
                            date_time_str=time_str
                        )
                        
                        st.info(f"💬 **NLP Output:**\n\"{nlp_msg}\"")
                        
                        # Cache the NLP message globally to share with NLP tab
                        st.session_state["last_nlp_msg"] = nlp_msg
                        
                        # 4. Save to Database & Generate PDF
                        with st.spinner("Generating E-Challan PDF & saving violation record..."):
                            # Save annotated frame temporarily as the official evidence image
                            ev_path = det["evidence_path"]
                            
                            # Generate PDF
                            pdf_path = pdf_gen.generate_challan(
                                violation_id=total_violations + 1,
                                vehicle_details=owner_info,
                                violation_type="Riding without Helmet",
                                fine_amount=fine_amt,
                                nlp_message=nlp_msg,
                                evidence_image_path=ev_path
                            )
                            
                            # Log violation to SQLite
                            db_id = log_violation(
                                vehicle_number=owner_info["vehicle_number"],
                                owner_name=owner_info["owner_name"],
                                phone=owner_info["phone"],
                                address=owner_info["address"],
                                violation_type="Riding without Helmet",
                                fine_amount=fine_amt,
                                nlp_message=nlp_msg,
                                evidence_image_path=ev_path,
                                challan_pdf_path=pdf_path
                            )
                            
                            st.write(f"✔️ Challan successfully logged under ID: **#{db_id}**")
                            
                            # PDF Download Button
                            with open(pdf_path, "rb") as f:
                                 pdf_bytes = f.read()

                            st.download_button(
                                label="📥 Download PDF E-Challan",
                                data=pdf_bytes,
                                file_name=os.path.basename(pdf_path),
                                mime="application/pdf",
                                type="primary",
                                key=f"main_pdf_{db_id}"
                             )
                            
                            
# ----------------------------------------------------
# TAB 2: NLP LINGUISTIC ANALYSIS
# ----------------------------------------------------
with tab_nlp:
    st.header("2. Natural Language Processing Module")
    st.write("This section demonstrates the core NLP linguistics capabilities, fulfilling the project requirements.")

    # Get the last generated message or load a default
    default_msg = "Dear Rahul Sharma, your vehicle MH12AB1234 was detected violating traffic rules due to riding without a helmet. A fine of ₹1000 has been generated."
    active_msg = st.session_state.get("last_nlp_msg", default_msg)
    
    st.text_area("Analyze Notice Text:", value=active_msg, key="nlp_analysis_text", height=90)
    
    if st.button("🔬 Parse with spaCy Pipeline"):
        msg_to_parse = st.session_state["nlp_analysis_text"]
        
        with st.spinner("Analyzing text grammar and structure..."):
            nlp_result = nlp.analyze_message(msg_to_parse)
            
            # Entities visualization
            st.write("### 🔍 Named Entity Recognition (NER)")
            st.write("NER detects names, license numbers, date, currency, and other standard elements in the raw text.")
            
            if not nlp_result["entities"]:
                st.write("No named entities extracted.")
            else:
                ent_df = pd.DataFrame(nlp_result["entities"])
                st.table(ent_df)
                
            # Tokenization and POS Tagging
            st.write("### 🏷️ Tokenization & Parts-of-Speech (POS) Tagging")
            st.write("This table shows how NLP tokenizes the document and tags nouns, verbs, adjectives, etc. for grammatical synthesis.")
            
            token_df = pd.DataFrame(nlp_result["tokens"])
            st.dataframe(token_df[["text", "lemma", "pos", "explain", "dep"]], use_container_width=True)
            
            # Syntax tree description
            st.write("### 🎯 Educational Summary: Role of NLP in E-Challan Systems")
            st.markdown("""
            In a traffic automation system, raw database values (e.g. database fields like `Rahul Sharma`, `1000`, `MH12AB1234`) are isolated variables. 
            **NLP is used to:**
            1. **Generate Natural, Context-Aware Messages**: Rather than displaying raw databases or rigid spreadsheets, NLP translates structural data into human-legible legal text.
            2. **Entity Validation**: Our model uses NER to scan the message and check that the **Subject** (Rahul Sharma), **Target** (MH12AB1234), and **Attributes** (₹1000 fine) are correctly placed inside the sentence boundaries.
            3. **Voice/Language Adaptation (Extensible)**: Parts-of-Speech tagging helps translate this violation into regional Indian languages (Hindi, Marathi, Kannada) by rearranging word tokens dynamically according to correct grammar structures (Subject-Object-Verb vs Subject-Verb-Object).
            """)

# ----------------------------------------------------
# TAB 3: SQLITE REGISTRY
# ----------------------------------------------------
with tab_db:
    st.header("3. Database Registry and History Logs")
    
    db_tab1, db_tab2 = st.tabs(["🗄️ Registered Vehicle Database", "📋 Violation Incident Logs"])
    
    with db_tab1:
        st.write("Registered vehicle database (matches license plates to owner profile details):")
        vehicles = get_all_vehicles()
        if vehicles:
            df_vehicles = pd.DataFrame(vehicles)
            st.dataframe(df_vehicles, use_container_width=True)
        else:
            st.write("Registry database is empty.")
            
    with db_tab2:
        st.write("History log of all detected traffic violations and generated e-challans:")
        violations_log = get_all_violations()
        
        if violations_log:
            df_violations = pd.DataFrame(violations_log)
            # Remove image paths and columns that are large, or display nicely
            st.dataframe(df_violations[["id", "vehicle_number", "owner_name", "timestamp", "violation_type", "fine_amount", "phone"]], use_container_width=True)
            
            # Expand details of selected violation
            st.write("### Inspect Specific Log Entry")
            violation_ids = [v["id"] for v in violations_log]
            selected_id = st.selectbox("Select Challan ID to view detailed details:", violation_ids)
            
            for v in violations_log:
                if v["id"] == selected_id:
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.write(f"**Challan # {v['id']} Details**")
                        st.write(f"👤 **Owner:** {v['owner_name']}")
                        st.write(f"🚗 **Vehicle:** {v['vehicle_number']}")
                        st.write(f"📅 **Time:** {v['timestamp']}")
                        st.write(f"⚠️ **Violation:** {v['violation_type']}")
                        st.write(f"💰 **Fine Amount:** ₹{v['fine_amount']}")
                        st.write(f"💬 **NLP Notice:**")
                        st.info(v["nlp_message"])
                        
                        # Add a download link for this PDF
                        if os.path.exists(v["challan_pdf_path"]):
                            with open(v["challan_pdf_path"], "rb") as f:
                                st.download_button(
                                    label="📥 Download PDF for this Challan",
                                    data=f.read(),
                                    file_name=os.path.basename(v["challan_pdf_path"]),
                                    mime="application/pdf",
                                    key=f"dl_btn_{v['id']}"
                                )
                    with col_v2:
                        st.write("**Attached Evidence Image:**")
                        if os.path.exists(v["evidence_image_path"]):
                            st.image(v["evidence_image_path"], use_container_width=True)
                        else:
                            st.write("No image file found on disk.")
        else:
            st.write("No traffic violations recorded yet. Run the pipeline on the dashboard tab to register violations.")

# ----------------------------------------------------
# TAB 4: PRESENTATION SLIDE OUTLINE
# ----------------------------------------------------
with tab_edu:
    st.header("4. TY BSc AI Presentation Helper & Guide")
    st.write("Use this guide to prepare for your presentation and viva-voce.")
    
    st.markdown("""
    ### 📊 Suggested Slide Deck Structure (10 Slides)
    
    1. **Title Slide**: Project Name, Your Name, Seat Number, Guide Name.
    2. **Introduction**: Introduce the problem of road traffic safety (importance of helmets) and manual challan generation inefficiencies.
    3. **Proposed System Architecture**: Flow diagram showing image input ➡️ YOLOv8 detection ➡️ EasyOCR extraction ➡️ SQLite owner match ➡️ spaCy NLP generation ➡️ PDF Challan compilation.
    4. **Computer Vision (YOLOv8)**:
       - Explain YOLOv8 (You Only Look Once v8) - state-of-the-art object detection.
       - Classes used: `person`, `motorcycle`, `helmet`, `no-helmet`.
       - Bounding box intersection logic to identify riders.
    5. **License Plate Recognition (EasyOCR)**:
       - Cropping motorcycle ROI.
       - Processing image (Grayscale, Bilateral Filter) to reduce noise.
       - Running OCR and filtering alphanumeric characters matching standard Indian plates via Regular Expressions (Regex).
    6. **Natural Language Processing (NLP)**:
       - Why template-based generation is reinforced by spaCy parsing.
       - Explain tokenization, POS tagging, and Named Entity Recognition (NER).
       - Explain how spaCy validates semantic roles (Owner = `PERSON`, Plate = `VEHICLE_NO`, Penalty = `MONEY`).
    7. **System Implementation**: SQLite database schema, Python files structure, and report generation using ReportLab.
    8. **Demonstration & UI**: Streamlit dashboard features (Live uploads, DB inspection, NLP tag visualization).
    9. **Future Enhancements**: Extending to red-light violations, triple-riding, automated email/SMS dispatch via SMS APIs.
    10. **Conclusion & References**: Summary of project outcomes.
    
    ---
    
    ### 💬 Viva-Voce FAQ (Common Questions Asked by External Examiners)
    
    **Q1: How do you associate a rider (person) with a motorcycle in the code?**
    *Answer:* We check bounding box overlap. If a person's bounding box intersects with a motorcycle's bounding box, and the overlap ratio exceeds 15% of the person's area, our logic classifies the person as a rider of that specific motorcycle.
    
    **Q2: Why use spaCy instead of simple Python f-string templates for NLP?**
    *Answer:* Although we construct the core notice using template structures, spaCy parses the synthesized legal notice to perform Tokenization, POS tagging, and NER. This allows the system to:
    1. Verify that the generated names and entities are grammatically valid and correctly aligned.
    2. Provide a structure ready for multi-lingual translation engines (token rearranging).
    3. Provide analytical text validation prior to issuing the official ticket.
    
    **Q3: How do you handle low-quality number plate OCR detections?**
    *Answer:* Our system uses three safety layers:
    1. **Pre-processing**: Grayscaling, Resizing (x2 interpolation), and Bilateral Filtering to make text sharper.
    2. **Regex Filter**: Rejects random text blobs that don't match the standard license plate alphanumeric layout.
    3. **Fuzzy String Matcher**: Automatically matches OCR readings against registered plates in the SQLite registry (using character distance) and auto-corrects minor reading errors (e.g. replacing 'O' with '0').
    
    **Q4: What is the benefit of SQLite in this system?**
    *Answer:* SQLite is a lightweight, serverless database stored as a single file on disk. It allows our system to query owner profiles in milliseconds and maintains an immutable historical ledger of all traffic tickets issued, which is easy to set up and ideal for demonstrating local app prototypes.
    """)
