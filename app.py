import streamlit as st
import os
import cv2
import pandas as pd
from datetime import datetime

# Import project modules
from database import (
    init_db,
    get_vehicle_owner,
    log_violation,
    get_all_violations,
    get_all_vehicles,
    add_vehicle_owner
)
from detector import TrafficViolationDetector
from ocr_module import LicensePlateRecognizer
from nlp_engine import TrafficNLP
from challan_generator import ChallanPDFGenerator


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Traffic Violation & E-Challan System",
    page_icon="🚥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }

    .stApp {
        background-color: #0F172A;
    }

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


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_media")
os.makedirs(SAMPLE_DIR, exist_ok=True)


# ============================================================
# RESOURCE LOADING
# ============================================================

# IMPORTANT:
# The heavy YOLO detector is NOT loaded during initial page load.
# It is loaded only when the user clicks the detection button.
#
# This prevents Streamlit Cloud from getting stuck at:
# "Running load_resources"
#
# EasyOCR is also lazy-loaded only when detection runs.


@st.cache_resource
def load_basic_resources():
    """
    Load lightweight resources required by the dashboard.

    Heavy computer-vision models are intentionally excluded.
    """
    init_db()

    nlp = TrafficNLP()
    pdf = ChallanPDFGenerator()

    return nlp, pdf


@st.cache_resource
def load_detector():
    """
    Lazy-load the complete YOLO traffic detector.

    This is intentionally called only when the user runs
    the AI Detection Pipeline.
    """
    return TrafficViolationDetector()


@st.cache_resource
def load_ocr():
    """
    Lazy-load EasyOCR only when the detection pipeline needs it.
    """
    return LicensePlateRecognizer()


# Load only lightweight resources at startup.
try:

    nlp, pdf_gen = load_basic_resources()

except Exception as e:

    st.error(
        f"Error loading core system components: {e}"
    )

    st.stop()


# Detector starts as unavailable.
# It will be initialized only when detection is requested.

detector = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("### 🚥 System Dashboard")
st.sidebar.info("TY BSc AI Project - Version 1.0")

st.sidebar.markdown("### ⚙️ Model Parameters")

conf_val = st.sidebar.slider(
    "YOLOv8 BBox Confidence",
    0.10,
    0.90,
    0.25,
    0.05
)

ocr_sim_correct = st.sidebar.checkbox(
    "Enable OCR Fuzzy Matcher",
    value=True
)


# ============================================================
# REGISTER NEW VEHICLE
# ============================================================

with st.sidebar.expander("📝 Register New Vehicle (Registry)"):

    new_plate = st.text_input(
        "Number Plate",
        placeholder="MH12AB1234"
    )

    new_owner = st.text_input(
        "Owner Name",
        placeholder="Rahul Sharma"
    )

    new_phone = st.text_input(
        "Mobile Number",
        placeholder="+91 9999999999"
    )

    new_addr = st.text_area(
        "Owner Address",
        placeholder="Street name, City, State"
    )

    if st.button("Register Vehicle"):

        if new_plate and new_owner and new_phone and new_addr:

            add_vehicle_owner(
                new_plate,
                new_owner,
                new_phone,
                new_addr
            )

            st.sidebar.success(
                f"Vehicle {new_plate} registered successfully!"
            )

            st.rerun()

        else:

            st.sidebar.error(
                "All registry fields are mandatory."
            )


# ============================================================
# MAIN APPLICATION BANNER
# ============================================================

st.markdown("""
<div class="title-banner">
    <h1>Traffic Violation Detector & E-Challan Generator</h1>
    <p>
        Integrated Computer Vision (YOLOv8 + EasyOCR)
        & NLP (spaCy Parser) System for Traffic Safety Enforcement
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# METRIC SUMMARY
# ============================================================

v_list = get_all_violations()

total_violations = len(v_list)
reg_vehicles = len(get_all_vehicles())

col_m1, col_m2, col_m3, col_m4 = st.columns(4)


with col_m1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{total_violations}</div>
            <div class="metric-label">Total Violations Logged</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col_m2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{reg_vehicles}</div>
            <div class="metric-label">Registered Vehicles</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col_m3:

    pending_fines = total_violations * 1000

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">₹{pending_fines}</div>
            <div class="metric-label">Fines Generated</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col_m4:

    # Detector is intentionally not loaded during startup.
    # The service status therefore reflects whether the detector
    # has already been initialized in this session.

    detector_status = (
        "Ready"
        if detector is not None
        else "Standby"
    )

    detector_color = (
        "#22C55E"
        if detector is not None
        else "#3B82F6"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value"
                 style="color: {detector_color};">
                {detector_status}
            </div>
            <div class="metric-label">
                YOLOv8 Service Status
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# TABS
# ============================================================

tab_dash, tab_nlp, tab_db, tab_edu = st.tabs([
    "🚥 Traffic Violation Detection",
    "📝 NLP & Linguistic Analysis",
    "🗄️ SQLite Database Registry",
    "🎓 BSc Project Presentation Help"
])


# ============================================================
# TAB 1: DETECTION DASHBOARD
# ============================================================

with tab_dash:

    st.header("1. Input Traffic Media")

    col_input, col_opt = st.columns([2, 1])

    with col_input:

        upload_mode = st.radio(
            "Select input source:",
            [
                "Use Sample Image (No Helmet Demo)",
                "Upload Custom Traffic Image"
            ],
            horizontal=True
        )

    img_path = None


    # --------------------------------------------------------
    # SAMPLE IMAGE
    # --------------------------------------------------------

    if upload_mode == "Use Sample Image (No Helmet Demo)":

        sample_path = os.path.join(
            SAMPLE_DIR,
            "traffic_sample.jpg"
        )

        if os.path.exists(sample_path):

            img_path = sample_path

            st.success(
                "Sample image loaded. Click 'Run Detection Pipeline' below."
            )

        else:

            st.error(
                "Sample image not found. Please upload a custom image."
            )


    # --------------------------------------------------------
    # CUSTOM IMAGE
    # --------------------------------------------------------

    else:

        uploaded_file = st.file_uploader(
            "Upload traffic JPG/PNG image",
            type=["jpg", "png", "jpeg"]
        )

        if uploaded_file:

            temp_path = os.path.join(
                SAMPLE_DIR,
                "temp_upload.jpg"
            )

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            img_path = temp_path

            st.image(
                img_path,
                caption="Uploaded Image",
                width=400
            )


    # --------------------------------------------------------
    # RUN DETECTION
    # --------------------------------------------------------

    if img_path:

        run_btn = st.button(
            "🚀 Run AI Detection Pipeline",
            type="primary",
            use_container_width=True
        )

        if run_btn:

            # ------------------------------------------------
            # STEP 1: LOAD YOLO DETECTOR ONLY NOW
            # ------------------------------------------------

            with st.spinner(
                "Loading YOLOv8 traffic detection models..."
            ):

                try:

                    detector = load_detector()

                except Exception as e:

                    st.error(
                        f"YOLO detector could not be loaded: {e}"
                    )

                    st.stop()


            # ------------------------------------------------
            # STEP 2: LOAD OCR ONLY NOW
            # ------------------------------------------------

            with st.spinner(
                "Loading EasyOCR model for license-plate recognition..."
            ):

                try:

                    ocr = load_ocr()

                except Exception as e:

                    st.error(
                        f"EasyOCR could not be loaded: {e}"
                    )

                    st.stop()


            # ------------------------------------------------
            # STEP 3: COMPUTER VISION
            # ------------------------------------------------

            with st.spinner(
                "Processing Computer Vision Pipeline..."
            ):

                annotated_img, detections = detector.detect_violations(
                    img_path
                )


            # Convert BGR to RGB
            annotated_img_rgb = cv2.cvtColor(
                annotated_img,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # SHOW RESULTS
            # ------------------------------------------------

            st.write("### AI Bounding Box Annotations")

            col_res1, col_res2 = st.columns(2)


            with col_res1:

                orig_img = cv2.imread(img_path)

                orig_img_rgb = cv2.cvtColor(
                    orig_img,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    orig_img_rgb,
                    caption="Input Frame / Raw Image",
                    use_container_width=True
                )


            with col_res2:

                st.image(
                    annotated_img_rgb,
                    caption=(
                        "YOLOv8 Analysis "
                        "(Vehicles, Riders, Helmet Safety Status)"
                    ),
                    use_container_width=True
                )


            # ------------------------------------------------
            # NO VIOLATIONS
            # ------------------------------------------------

            if not detections:

                st.markdown(
                    """
                    <div class="success-alert">
                        <div class="violation-title"
                             style="color: #22C55E;">
                            ✔️ Safety Compliance Verified
                        </div>

                        No traffic violations detected in this frame.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # VIOLATIONS FOUND
            # ------------------------------------------------

            else:

                for idx, det in enumerate(detections):

                    violation_type_raw = det.get(
                        "violation_type",
                        "Riding without Helmet"
                    )

                    violation_type_lower = (
                        violation_type_raw.lower()
                    )


                    # Determine violation label
                    if "triple" in violation_type_lower:

                        display_violation = "Triple Riding"

                        violation_description = (
                            "Three riders detected on the same motorcycle."
                        )

                        nlp_violation = "triple riding"

                    else:

                        display_violation = "Riding Without Helmet"

                        violation_description = (
                            "Motorcycle rider detected without a helmet."
                        )

                        nlp_violation = "riding without a helmet"


                    # ------------------------------------------------
                    # VIOLATION ALERT
                    # ------------------------------------------------

                    st.markdown(
                        f"""
                        <div class="violation-alert">
                            <div class="violation-title">
                                🚨 Traffic Violation Alert:
                                {display_violation}
                            </div>

                            {violation_description}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    col_det1, col_det2 = st.columns([1, 2])


                    # ------------------------------------------------
                    # MOTORCYCLE EVIDENCE
                    # ------------------------------------------------

                    with col_det1:

                        st.write(
                            "**Evidence Snippet (Motorcycle ROI):**"
                        )

                        crop = det.get(
                            "motorcycle_crop"
                        )

                        if crop is not None:

                            crop_rgb = cv2.cvtColor(
                                crop,
                                cv2.COLOR_BGR2RGB
                            )

                            st.image(
                                crop_rgb,
                                use_container_width=True,
                                caption="Cropped Vehicle ROI"
                            )

                        else:

                            st.warning(
                                "Motorcycle crop not available."
                            )


                    # ------------------------------------------------
                    # OCR + DATABASE
                    # ------------------------------------------------

                    with col_det2:

                        st.write(
                            "🔍 **License Plate Extraction (EasyOCR):**"
                        )


                        raw_plate, confidence = (
                            ocr.extract_license_plate(
                                det["motorcycle_crop"]
                            )
                        )


                        # Registered plates
                        all_reg_plates = [
                            v["vehicle_number"]
                            for v in get_all_vehicles()
                        ]


                        # Fuzzy matching
                        if ocr_sim_correct:

                            matched_plate, match_conf = (
                                ocr.fuzzy_match_plate(
                                    raw_plate,
                                    all_reg_plates
                                )
                            )

                        else:

                            matched_plate = raw_plate
                            match_conf = 1.0


                        # ------------------------------------------------
                        # OCR SUCCESS
                        # ------------------------------------------------

                        if matched_plate:

                            st.success(
                                f"Number Plate Recognized: "
                                f"**{matched_plate}** "
                                f"(OCR Raw: `{raw_plate}`, "
                                f"Conf: {confidence:.2f})"
                            )


                        # ------------------------------------------------
                        # OCR FALLBACK
                        # ------------------------------------------------

                        else:

                            matched_plate = "MH12AB1234"

                            st.warning(
                                "OCR reading low. "
                                "Fallback assigned for presentation demo: "
                                f"**{matched_plate}**"
                            )


                        # ------------------------------------------------
                        # DATABASE LOOKUP
                        # ------------------------------------------------

                        owner_info = get_vehicle_owner(
                            matched_plate
                        )


                        if owner_info:

                            st.markdown(
                                "🗄️ **SQLite Owner Record Found:**"
                            )

                            st.json(owner_info)


                        else:

                            owner_info = {
                                "vehicle_number": matched_plate,
                                "owner_name": "Rahul Sharma",
                                "phone": "+91 98765 43210",
                                "address": (
                                    "Flat 402, Sunshine Apartments, "
                                    "Pune, Maharashtra - 411001"
                                )
                            }

                            st.warning(
                                "License plate not in registry database. "
                                "Simulating default registration details."
                            )

                            st.json(owner_info)


                    st.write("---")


                    # ------------------------------------------------
                    # NLP MESSAGE
                    # ------------------------------------------------

                    st.write(
                        "⚙️ **NLP Notice Message Generation (spaCy):**"
                    )


                    fine_amt = 1000

                    time_str = datetime.now().strftime(
                        "%Y-%m-%d at %I:%M %p"
                    )


                    nlp_msg = nlp.generate_violation_message(
                        owner_name=owner_info["owner_name"],
                        vehicle_number=owner_info["vehicle_number"],
                        violation_type=nlp_violation,
                        fine_amount=fine_amt,
                        date_time_str=time_str
                    )


                    st.info(
                        f'💬 **NLP Output:**\n"{nlp_msg}"'
                    )


                    # Save NLP message
                    st.session_state["last_nlp_msg"] = nlp_msg


                    # ------------------------------------------------
                    # SAVE DATABASE + PDF
                    # ------------------------------------------------

                    with st.spinner(
                        "Generating E-Challan PDF & saving violation record..."
                    ):

                        ev_path = det.get(
                            "evidence_path"
                        )


                        # Generate PDF
                        pdf_path = pdf_gen.generate_challan(
                            violation_id=total_violations + idx + 1,
                            vehicle_details=owner_info,
                            violation_type=display_violation,
                            fine_amount=fine_amt,
                            nlp_message=nlp_msg,
                            evidence_image_path=ev_path
                        )


                        # Log violation
                        db_id = log_violation(
                            vehicle_number=owner_info[
                                "vehicle_number"
                            ],

                            owner_name=owner_info[
                                "owner_name"
                            ],

                            phone=owner_info[
                                "phone"
                            ],

                            address=owner_info[
                                "address"
                            ],

                            violation_type=display_violation,

                            fine_amount=fine_amt,

                            nlp_message=nlp_msg,

                            evidence_image_path=ev_path,

                            challan_pdf_path=pdf_path
                        )


                        st.write(
                            f"✔️ Challan successfully logged under "
                            f"ID: **#{db_id}**"
                        )


                        # ------------------------------------------------
                        # PDF DOWNLOAD
                        # ------------------------------------------------

                        with open(
                            pdf_path,
                            "rb"
                        ) as f:

                            pdf_bytes = f.read()


                        st.download_button(
                            label="📥 Download PDF E-Challan",
                            data=pdf_bytes,
                            file_name=os.path.basename(
                                pdf_path
                            ),
                            mime="application/pdf",
                            type="primary",
                            key=f"main_pdf_{db_id}"
                        )


# ============================================================
# TAB 2: NLP LINGUISTIC ANALYSIS
# ============================================================

with tab_nlp:

    st.header(
        "2. Natural Language Processing Module"
    )

    st.write(
        "This section demonstrates the core NLP linguistics "
        "capabilities, fulfilling the project requirements."
    )


    default_msg = (
        "Dear Rahul Sharma, your vehicle MH12AB1234 "
        "was detected violating traffic rules due to "
        "riding without a helmet. A fine of ₹1000 has been generated."
    )


    active_msg = st.session_state.get(
        "last_nlp_msg",
        default_msg
    )


    st.text_area(
        "Analyze Notice Text:",
        value=active_msg,
        key="nlp_analysis_text",
        height=90
    )


    if st.button(
        "🔬 Parse with spaCy Pipeline"
    ):

        msg_to_parse = st.session_state[
            "nlp_analysis_text"
        ]


        with st.spinner(
            "Analyzing text grammar and structure..."
        ):

            nlp_result = nlp.analyze_message(
                msg_to_parse
            )


            # ------------------------------------------------
            # NER
            # ------------------------------------------------

            st.write(
                "### 🔍 Named Entity Recognition (NER)"
            )

            st.write(
                "NER detects names, license numbers, date, "
                "currency, and other standard elements in the raw text."
            )


            if not nlp_result["entities"]:

                st.write(
                    "No named entities extracted."
                )

            else:

                ent_df = pd.DataFrame(
                    nlp_result["entities"]
                )

                st.table(ent_df)


            # ------------------------------------------------
            # TOKENIZATION + POS
            # ------------------------------------------------

            st.write(
                "### 🏷️ Tokenization & Parts-of-Speech (POS) Tagging"
            )

            st.write(
                "This table shows how NLP tokenizes the document "
                "and tags nouns, verbs, adjectives, etc."
            )


            token_df = pd.DataFrame(
                nlp_result["tokens"]
            )


            st.dataframe(
                token_df[
                    [
                        "text",
                        "lemma",
                        "pos",
                        "explain",
                        "dep"
                    ]
                ],
                use_container_width=True
            )


            # ------------------------------------------------
            # EDUCATIONAL SUMMARY
            # ------------------------------------------------

            st.write(
                "### 🎯 Educational Summary: "
                "Role of NLP in E-Challan Systems"
            )


            st.markdown(
                """
                In a traffic automation system, raw database
                values such as owner name, vehicle number and
                fine amount are isolated variables.

                **NLP is used to:**

                1. **Generate Natural, Context-Aware Messages**:
                   NLP converts structured traffic data into
                   human-readable legal notification text.

                2. **Entity Validation**:
                   NER can scan the generated message and identify
                   important entities such as the owner and vehicle.

                3. **Language Adaptation**:
                   POS tagging and linguistic analysis provide a
                   foundation for future multilingual traffic notices.
                """
            )


# ============================================================
# TAB 3: SQLITE DATABASE REGISTRY
# ============================================================

with tab_db:

    st.header(
        "3. Database Registry and History Logs"
    )


    db_tab1, db_tab2 = st.tabs(
        [
            "🗄️ Registered Vehicle Database",
            "📋 Violation Incident Logs"
        ]
    )


    # --------------------------------------------------------
    # REGISTERED VEHICLES
    # --------------------------------------------------------

    with db_tab1:

        st.write(
            "Registered vehicle database "
            "(matches license plates to owner profiles):"
        )


        vehicles = get_all_vehicles()


        if vehicles:

            df_vehicles = pd.DataFrame(
                vehicles
            )

            st.dataframe(
                df_vehicles,
                use_container_width=True
            )

        else:

            st.write(
                "Registry database is empty."
            )


    # --------------------------------------------------------
    # VIOLATION HISTORY
    # --------------------------------------------------------

    with db_tab2:

        st.write(
            "History log of all detected traffic violations "
            "and generated e-challans:"
        )


        violations_log = get_all_violations()


        if violations_log:

            df_violations = pd.DataFrame(
                violations_log
            )


            st.dataframe(
                df_violations[
                    [
                        "id",
                        "vehicle_number",
                        "owner_name",
                        "timestamp",
                        "violation_type",
                        "fine_amount",
                        "phone"
                    ]
                ],
                use_container_width=True
            )


            st.write(
                "### Inspect Specific Log Entry"
            )


            violation_ids = [
                v["id"]
                for v in violations_log
            ]


            selected_id = st.selectbox(
                "Select Challan ID to view detailed details:",
                violation_ids
            )


            for v in violations_log:

                if v["id"] == selected_id:

                    col_v1, col_v2 = st.columns(2)


                    with col_v1:

                        st.write(
                            f"**Challan # {v['id']} Details**"
                        )

                        st.write(
                            f"👤 **Owner:** {v['owner_name']}"
                        )

                        st.write(
                            f"🚗 **Vehicle:** {v['vehicle_number']}"
                        )

                        st.write(
                            f"📅 **Time:** {v['timestamp']}"
                        )

                        st.write(
                            f"⚠️ **Violation:** {v['violation_type']}"
                        )

                        st.write(
                            f"💰 **Fine Amount:** ₹{v['fine_amount']}"
                        )

                        st.write(
                            "💬 **NLP Notice:**"
                        )

                        st.info(
                            v["nlp_message"]
                        )


                        if os.path.exists(
                            v["challan_pdf_path"]
                        ):

                            with open(
                                v["challan_pdf_path"],
                                "rb"
                            ) as f:

                                st.download_button(
                                    label=(
                                        "📥 Download PDF "
                                        "for this Challan"
                                    ),

                                    data=f.read(),

                                    file_name=os.path.basename(
                                        v["challan_pdf_path"]
                                    ),

                                    mime="application/pdf",

                                    key=f"dl_btn_{v['id']}"
                                )


                    with col_v2:

                        st.write(
                            "**Attached Evidence Image:**"
                        )


                        if os.path.exists(
                            v["evidence_image_path"]
                        ):

                            st.image(
                                v["evidence_image_path"],
                                use_container_width=True
                            )

                        else:

                            st.write(
                                "No image file found on disk."
                            )


        else:

            st.write(
                "No traffic violations recorded yet. "
                "Run the pipeline on the dashboard tab "
                "to register violations."
            )


# ============================================================
# TAB 4: PRESENTATION HELP
# ============================================================

with tab_edu:

    st.header(
        "4. TY BSc AI Presentation Helper & Guide"
    )

    st.write(
        "Use this guide to prepare for your presentation."
    )


    st.markdown(
        """
        ### 📊 Suggested Slide Deck Structure (10 Slides)

        1. **Title Slide**:
           Project Name, Your Name, Seat Number, Guide Name.

        2. **Introduction**:
           Explain the problem of road traffic safety,
           helmet violations and automated challan generation.

        3. **Proposed System Architecture**:
           Image Input → YOLOv8 Detection → EasyOCR →
           SQLite Owner Match → spaCy NLP → PDF Challan.

        4. **Computer Vision (YOLOv8)**:
           - Object detection
           - Person and motorcycle detection
           - Helmet / No Helmet detection
           - Rider-to-motorcycle association
           - Triple Riding detection

        5. **License Plate Recognition (EasyOCR)**:
           - Motorcycle ROI extraction
           - Image preprocessing
           - OCR extraction
           - Regex filtering
           - Fuzzy matching against registered plates

        6. **Natural Language Processing (NLP)**:
           - Template-based notice generation
           - Tokenization
           - POS tagging
           - Named Entity Recognition
           - spaCy linguistic analysis

        7. **System Implementation**:
           SQLite database, Python modules and ReportLab PDF generation.

        8. **Demonstration & UI**:
           Streamlit dashboard, traffic image upload,
           detection results, database and challan generation.

        9. **Future Enhancements**:
           Improved detection accuracy, additional traffic
           violations and automated notification systems.

        10. **Conclusion & References**:
            Summary of the implemented system and references.
        """
    )