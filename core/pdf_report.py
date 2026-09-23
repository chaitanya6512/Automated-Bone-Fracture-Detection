from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage
import numpy as np

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Table
from core.preprocessing import preprocess_visual_image


def generate_pdf_report(
    save_path,
    original_image_path,
    gradcam_image_path,
    body_part,
    body_conf,
    fracture_result,
    fracture_conf,
    is_uncertain=False
):

    doc = SimpleDocTemplate(
        save_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    elements = []

    # Title
    title = Paragraph(
        "<b>Bone Fracture Detection Report (Grad-CAM++)</b>",
        styles["Title"]
    )
    elements.append(title)
    elements.append(Spacer(1, 0.3 * inch))

    # Timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp = Paragraph(
        f"<b>Generated:</b> {current_time}",
        styles["BodyText"]
    )
    elements.append(timestamp)
    elements.append(Spacer(1, 0.2 * inch))

    # Uncertainty Banner
    uncertainty_str = "<br/><font color='orange'><b>⚠️ Clinical Risk Alert:</b> Prediction confidence is below threshold. Radiologist evaluation required.</font>" if is_uncertain else ""

    details = Paragraph(
        f"""
        <b>Predicted Body Part:</b> {body_part} ({body_conf:.2f}%)<br/>
        <b>Fracture Status:</b> {fracture_result.capitalize()} ({fracture_conf:.2f}%){uncertainty_str}
        """,
        styles["BodyText"]
    )

    elements.append(details)
    elements.append(Spacer(1, 0.12 * inch))

    elements.append(
        Paragraph("<b>X-ray & Grad-CAM++ Analysis</b>", styles["Heading4"])
    )

    # Ensure ReportLab compatibility (convert DICOM / numpy array to temp PNG if needed)
    orig_path_str = str(original_image_path)
    temp_orig_path = original_image_path
    if orig_path_str.lower().endswith(('.dcm', '.dicom')):
        img_np = preprocess_visual_image(original_image_path)
        temp_orig_path = Path(save_path).parent / "temp_report_orig.png"
        PILImage.fromarray(img_np).save(temp_orig_path)

    original_img = RLImage(str(temp_orig_path))
    original_img.drawHeight = 2.4 * inch
    original_img.drawWidth = 2.4 * inch

    gradcam_img = RLImage(str(gradcam_image_path))
    gradcam_img.drawHeight = 2.4 * inch
    gradcam_img.drawWidth = 2.4 * inch

    image_table = Table([
        [original_img, gradcam_img]
    ])

    elements.append(image_table)
    elements.append(Spacer(1, 0.15 * inch))

    why_predicted = Paragraph(
        f"""
        <b>Why the Model Predicted This Result (Grad-CAM++ Explainability):</b><br/><br/>
        The Grad-CAM++ visualization highlights the spatial regions of interest
        in the X-ray image that influenced the CNN model's decision.<br/>
        Warmer regions (red/yellow) indicate areas where high-order feature maps
        focused strongly while detecting abnormalities.<br/>
        In this case, the highlighted regions suggest features associated with
        <b>{fracture_result}</b> findings in the {body_part.lower()} X-ray image.
        """,
        styles["BodyText"]
    )

    elements.append(why_predicted)
    elements.append(Spacer(1, 0.15 * inch))

    disclaimer = Paragraph(
        """
        <b>Disclaimer:</b><br/>
        This system is intended for educational and research assistance only.
        It should not replace professional medical diagnosis by a certified radiologist.
        """,
        styles["BodyText"]
    )

    elements.append(disclaimer)

    doc.build(elements)

    # Clean up temp file if created
    if orig_path_str.lower().endswith(('.dcm', '.dicom')) and Path(temp_orig_path).exists():
        try:
            Path(temp_orig_path).unlink()
        except Exception:
            pass

    return save_path