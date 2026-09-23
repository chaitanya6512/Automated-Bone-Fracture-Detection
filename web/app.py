import sys
from pathlib import Path

# Ensure project root is in sys.path so 'core' modules can be imported from any launch directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

from core.predictions import predict, get_model
from core.gradcam import generate_gradcam
from core.pdf_report import generate_pdf_report
from core.preprocessing import read_image_file

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def normalize_image_for_display(img: Image.Image, target_size=(320, 320)) -> Image.Image:
    """Resize image uniformly maintaining aspect ratio and centering to exact target_size."""
    return ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
icon_path = PROJECT_ROOT / "assets" / "logo.png"
if not icon_path.exists():
    icon_path = PROJECT_ROOT / "assets" / "icon.png"

st.set_page_config(
    page_title="Automated Bone Fracture Detection System",
    page_icon=str(icon_path) if icon_path.exists() else "🦴",
    layout="wide"
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp { background-color: #0e1117; }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --------------------------------------------------
# Performance Optimization: Model Caching
# --------------------------------------------------
@st.cache_resource
def preload_models():
    """Preload models into memory with st.cache_resource to prevent re-reading from disk."""
    loaded = []
    errors = []
    for model_name in ["Parts", "Elbow", "Hand", "Shoulder"]:
        try:
            get_model(model_name)
            loaded.append(model_name)
        except Exception as e:
            errors.append((model_name, str(e)))
    return loaded, errors

loaded_models, model_errors = preload_models()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("🦴 Automated Bone Fracture Detection AI Dashboard")
    st.markdown("---")

    st.subheader("Model Architecture")
    st.write("DenseNet121 CNN + Grad-CAM++")

    st.subheader("Supported Formats")
    st.write("✔ Standard: PNG, JPG, JPEG")
    st.write("✔ Medical DICOM: `.dcm`, `.dicom` (pydicom)")

    st.subheader("Supported Body Parts")
    st.write("• Elbow\n• Hand\n• Shoulder")

    st.markdown("---")
    if model_errors:
        st.warning(f"⚠️ {len(model_errors)} Model(s) Failed to Load")
        for m_name, err in model_errors:
            st.error(f"**{m_name}**: {err}")
    else:
        st.success("⚡ AI Models Cached & Ready")
    st.markdown("---")

    st.warning("⚠️ Research & Diagnostic Support System. Always consult a certified radiologist.")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#00BFFF; font-size:40px;'>
        Automated Bone Fracture Detection System
    </h1>
    <p style='text-align:center; font-size:17px; color:lightgray;'>
        AI-Powered Musculoskeletal X-Ray Analysis with Grad-CAM++ Explainability & DICOM Support
    </p>
    """,
    unsafe_allow_html=True
)

# Output Directories
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# File Upload Section
# --------------------------------------------------
st.markdown("## 📥 Upload X-Ray Image(s)")
st.write("Upload one or multiple X-ray images (PNG, JPG, JPEG, DCM/DICOM) for analysis.")

uploaded_files = st.file_uploader(
    "Browse or drag X-ray file(s):",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
    accept_multiple_files=True
)

if uploaded_files:
    saved_file_paths = []
    processed_images = []
    normalized_previews = []

    for idx, file_obj in enumerate(uploaded_files):
        save_path = UPLOAD_DIR / file_obj.name
        with open(save_path, "wb") as f:
            f.write(file_obj.getbuffer())
        saved_file_paths.append(save_path)

        # Read RGB via DICOM/PIL wrapper
        raw_rgb = read_image_file(str(save_path))
        pil_img = Image.fromarray(raw_rgb)
        processed_images.append(pil_img)

        # Create normalized thumbnail preview (300x300)
        norm_img = normalize_image_for_display(pil_img, target_size=(300, 300))
        normalized_previews.append(norm_img)

    # Normalized Preview Grid (Always 4 columns so single images stay small at 25% width)
    st.markdown("### 🖼️ Preview Uploaded Study")
    cols = st.columns(4)
    for idx, (norm_img, path_item) in enumerate(zip(normalized_previews, saved_file_paths)):
        with cols[idx % 4]:
            st.image(norm_img, caption=f"View #{idx+1}: {path_item.name}", use_column_width=True)

    st.markdown("---")

    # --------------------------------------------------
    # Analysis Trigger Button
    # --------------------------------------------------
    if st.button("🚀 Analyze Study", use_container_width=True):
        study_results = []

        with st.spinner("Analyzing X-ray views..."):
            for idx, file_path in enumerate(saved_file_paths):
                # Predict Body Part
                bone_type, bone_conf = predict(
                    str(file_path),
                    "Parts",
                    return_confidence=True
                )

                # Predict Fracture & Uncertainty
                pred_label, pred_conf, probs, is_uncertain = predict(
                    str(file_path),
                    bone_type,
                    return_uncertainty=True
                )

                # Generate Grad-CAM++ with unique filename per upload
                gradcam_out_path = REPORT_DIR.parent / "GradCAM" / f"gradcam_{file_path.stem}_{idx+1}.png"
                gradcam_out_path.parent.mkdir(parents=True, exist_ok=True)

                gradcam_path = generate_gradcam(
                    str(file_path),
                    bone_type,
                    output_path=str(gradcam_out_path)
                )

                # Load & normalize Grad-CAM image for UI display
                gradcam_pil = Image.open(gradcam_path)
                gradcam_norm = normalize_image_for_display(gradcam_pil, target_size=(250, 250))

                # Generate PDF Report for this file
                pdf_path = REPORT_DIR / f"Report_{file_path.stem}_View_{idx+1}.pdf"
                generate_pdf_report(
                    save_path=str(pdf_path),
                    original_image_path=str(file_path),
                    gradcam_image_path=str(gradcam_path),
                    body_part=bone_type,
                    body_conf=bone_conf,
                    fracture_result=pred_label,
                    fracture_conf=pred_conf,
                    is_uncertain=is_uncertain
                )

                # Store raw & normalized images for rendering
                orig_norm = normalize_image_for_display(processed_images[idx], target_size=(250, 250))

                study_results.append({
                    "file_path": file_path,
                    "body_part": bone_type,
                    "body_conf": bone_conf,
                    "fracture_result": pred_label,
                    "fracture_conf": pred_conf,
                    "is_uncertain": is_uncertain,
                    "gradcam_path": gradcam_path,
                    "gradcam_img": gradcam_norm,
                    "orig_img": orig_norm,
                    "raw_img": processed_images[idx],
                    "pdf_path": pdf_path
                })

        st.session_state["study_results"] = study_results
        st.success("✅ AI Analysis Complete!")

# --------------------------------------------------
# Main 3 Navigation Pages: Results, Visualization, Report
# --------------------------------------------------
if "study_results" in st.session_state and st.session_state["study_results"]:
    study_results = st.session_state["study_results"]

    st.markdown("---")
    
    # Primary 3 Navigation Tabs
    tab_results, tab_vis, tab_report = st.tabs([
        "📊 Results",
        "🔥 Visualization",
        "📄 Report"
    ])

    # ==================================================
    # PAGE 1: 📊 RESULTS
    # ==================================================
    with tab_results:
        st.markdown("## 📊 Clinical Results & Findings")

        any_fractured = any(r["fracture_result"] == "fractured" for r in study_results)
        any_uncertain = any(r["is_uncertain"] for r in study_results)

        if any_fractured:
            st.error("🚨 **OVERALL DIAGNOSIS: FRACTURE DETECTED** in one or more X-ray views.")
        else:
            st.success("✅ **OVERALL DIAGNOSIS: NORMAL BONE STRUCTURE** across all analyzed views.")

        if any_uncertain:
            st.warning("⚠️ **CLINICAL RISK WARNING:** One or more views exhibited prediction uncertainty (<65% confidence). Radiologist review recommended.")

        st.markdown("---")

        # Per-View Detailed Breakdowns
        view_tabs = st.tabs([f"View #{i+1} ({r['body_part']})" for i, r in enumerate(study_results)])
        for idx, (r, v_tab) in enumerate(zip(study_results, view_tabs)):
            with v_tab:
                col_info1, col_info2 = st.columns([2, 1])

                with col_info1:
                    st.markdown(f"### View #{idx+1}: `{r['file_path'].name}`")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Anatomical Region", r["body_part"], f"{r['body_conf']:.1f}% Confidence")
                    m2.metric("Fracture Status", r["fracture_result"].upper(), f"{r['fracture_conf']:.1f}% Confidence")
                    m3.metric("Clinical Risk", "UNCERTAIN" if r["is_uncertain"] else "CONFIDENT")

                    st.markdown("**Prediction Confidence:**")
                    st.progress(int(r["fracture_conf"]))

                    if r["is_uncertain"]:
                        st.warning(f"⚠️ Prediction confidence is {r['fracture_conf']:.1f}%. Elevated clinical uncertainty.")

                with col_info2:
                    st.image(r["orig_img"], caption=f"View #{idx+1} Thumbnail", use_column_width=True)

    # ==================================================
    # PAGE 2: 🔥 VISUALIZATION (Grad-CAM++)
    # ==================================================
    with tab_vis:
        st.markdown("### 🔥 Diagnostic Explainability (Grad-CAM++)")
        st.write("Visual explanations highlighting regions of interest that influenced the DenseNet121 model's decision.")

        # Interactive Controls (collapsed by default to maintain compact single-view layout)
        with st.expander("🛠️ Interactive Image Controls", expanded=False):
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                b_val = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1, key="vis_b")
            with col_ctrl2:
                c_val = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1, key="vis_c")
            with col_ctrl3:
                inv_col = st.checkbox("Invert Colors", key="vis_inv")

        # View Selector for Multi-View
        if len(study_results) > 1:
            selected_idx = st.selectbox(
                "Select X-Ray View to Inspect:",
                options=list(range(len(study_results))),
                format_func=lambda i: f"View #{i+1}: {study_results[i]['file_path'].name} ({study_results[i]['body_part']})"
            )
        else:
            selected_idx = 0

        res_item = study_results[selected_idx]

        # Apply user visual adjustments dynamically
        adjusted_img = res_item["raw_img"].copy()
        if b_val != 1.0:
            adjusted_img = ImageEnhance.Brightness(adjusted_img).enhance(b_val)
        if c_val != 1.0:
            adjusted_img = ImageEnhance.Contrast(adjusted_img).enhance(c_val)
        if inv_col:
            adjusted_img = ImageOps.invert(adjusted_img)

        # Normalize adjusted original and gradcam image to identical target size (250x250)
        norm_adjusted = normalize_image_for_display(adjusted_img, target_size=(250, 250))
        norm_gradcam = res_item["gradcam_img"]

        # Side-by-Side Comparison Grid (centered with constrained column ratio for clean single view)
        _, col_v1, _, col_v2, _ = st.columns([1, 2, 0.2, 2, 1])
        with col_v1:
            st.markdown("#### Original / Enhanced X-Ray")
            st.image(norm_adjusted, caption=f"Original ({res_item['body_part']})", use_column_width=True)

        with col_v2:
            st.markdown("#### Grad-CAM++ Spatial Heatmap")
            st.image(norm_gradcam, caption=f"Grad-CAM++ ({res_item['fracture_result'].upper()})", use_column_width=True)

        st.info(
            f"💡 **Grad-CAM++ Explainability Insight:** The heatmap highlights spatial features "
            f"that drove the model to predict **{res_item['fracture_result'].upper()}** for the {res_item['body_part']}. "
            f"Warmer (red/yellow) areas denote high-order feature activation."
        )

    # ==================================================
    # PAGE 3: 📄 REPORT
    # ==================================================
    with tab_report:
        st.markdown("## 📄 Diagnostic PDF Reports")
        st.write("Download formal PDF diagnostic reports with patient study metadata, Grad-CAM++ visualizations, and disclaimer notes.")

        for idx, r in enumerate(study_results):
            with st.container():
                st.markdown(f"### 📑 Diagnostic Report for View #{idx+1}: `{r['file_path'].name}`")

                r_col1, r_col2 = st.columns([3, 1])

                with r_col1:
                    st.write(f"• **Anatomical Part:** {r['body_part']} ({r['body_conf']:.1f}% confidence)")
                    st.write(f"• **Fracture Result:** {r['fracture_result'].capitalize()} ({r['fracture_conf']:.1f}% confidence)")
                    st.write(f"• **Clinical Status:** {'⚠️ UNCERTAIN (Radiologist Review Advised)' if r['is_uncertain'] else '✅ CONFIDENT'}")

                with r_col2:
                    if Path(r["pdf_path"]).exists():
                        with open(r["pdf_path"], "rb") as pdf_f:
                            st.download_button(
                                label=f"📥 Download PDF Report (View #{idx+1})",
                                data=pdf_f,
                                file_name=f"Bone_Fracture_Report_View_{idx+1}.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_page_{idx}"
                            )

                st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:13px; margin-top:30px;'>
        Automated Bone Fracture Detection System | DenseNet121, Grad-CAM++, pydicom & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
