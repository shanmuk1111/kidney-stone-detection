import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
import matplotlib.cm as cm
# pdf library
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus import HRFlowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import os
import random


# Initialize Flask app
app = Flask(__name__)

IMG_SIZE = 128

# Load trained model
model = load_model("../models/kidney_model.h5")

# VERY IMPORTANT: Build model once (fixes Grad-CAM error)
dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
model.predict(dummy_input)



def generate_pdf_report(label, confidence, risk, position,
                        area_percentage, size_category, warning,
                        filepath, heatmap_path):

    pdf_path = os.path.join("static", "Kidney_Stone_Diagnostic_Report.pdf")
    doc = SimpleDocTemplate(pdf_path)
    elements = []

    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading1'],
        textColor=colors.HexColor("#1f3c4d"),
        spaceAfter=14
    )

    normal_style = styles["Normal"]

    # Header
    elements.append(Paragraph("Kidney Stone AI Diagnostic Report", header_style))
    elements.append(HRFlowable(width="100%", thickness=1.2, color=colors.grey))
    elements.append(Spacer(1, 20))

    # Patient Info
    patient_id = f"KS-{random.randint(1000,9999)}"
    elements.append(Paragraph(f"<b>Patient ID:</b> {patient_id}", normal_style))
    elements.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%d-%m-%Y %H:%M')}", normal_style))
    elements.append(Spacer(1, 20))

    # 🔹 ADD IMAGES SECTION
    elements.append(Paragraph("<b>Imaging Analysis:</b>", styles["Heading3"]))
    elements.append(Spacer(1, 15))

    original_img = Image(filepath, width=2.8*inch, height=2.8*inch)
    heatmap_img = Image(heatmap_path, width=2.8*inch, height=2.8*inch)

    image_table = Table([[original_img, heatmap_img]], colWidths=[3*inch, 3*inch])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))

    elements.append(image_table)
    elements.append(Spacer(1, 25))

    # Risk color
    if risk == "High Risk":
        risk_color = colors.red
    elif risk == "Moderate Risk":
        risk_color = colors.orange
    else:
        risk_color = colors.green

    # Diagnostic Table
    data = [
        ["Parameter", "Result"],
        ["Prediction", label],
        ["Confidence", f"{confidence}%"],
        ["Risk Level", risk],
        ["Stone Position", position],
        ["Estimated Stone Area", f"{area_percentage}%"],
        ["Estimated Stone Size", size_category]
    ]

    table = Table(data, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('TEXTCOLOR', (1,3), (1,3), risk_color),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 25))

    # Clinical Interpretation
    elements.append(Paragraph("<b>Clinical Interpretation:</b>", styles["Heading3"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(warning, normal_style))
    elements.append(Spacer(1, 25))

    # Disclaimer
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "Disclaimer: This AI-assisted system provides supportive diagnostic analysis only. "
        "It does not replace professional medical evaluation.",
        normal_style
    ))

    doc.build(elements)

    return pdf_path




def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Stone Detected"
        confidence = prediction * 100
    else:
        label = "No Stone Detected"
        confidence = (1 - prediction) * 100

    return label, round(confidence, 2)


def generate_gradcam(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    last_conv_layer = model.get_layer("last_conv")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    
    # 🔥 SIZE ESTIMATION
    threshold = 180  # high activation threshold
    stone_pixels = np.sum(heatmap > threshold)
    total_pixels = heatmap.size

    area_percentage = (stone_pixels / total_pixels) * 100

    if area_percentage < 1:
        size_category = "Small"
    elif area_percentage < 3:
        size_category = "Medium"
    else:
        size_category = "Large"

    # 🔥 POSITION DETECTION LOGIC
    gray_heatmap = heatmap.copy()
    h, w = gray_heatmap.shape

    left_half = gray_heatmap[:, :w//2]
    right_half = gray_heatmap[:, w//2:]

    left_score = np.mean(left_half)
    right_score = np.mean(right_half)

    if left_score > right_score + 5:
        position = "Left Kidney"
    elif right_score > left_score + 5:
        position = "Right Kidney"
    else:
        position = "Central / Unclear"

    # Convert to color map
    heatmap_color = cm.jet(heatmap)[:, :, :3]
    heatmap_color = np.uint8(255 * heatmap_color)

    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

    superimposed_img = heatmap_color * 0.4 + original_img
    superimposed_img = np.uint8(superimposed_img)

    heatmap_filename = "gradcam_" + os.path.basename(img_path)
    heatmap_path = os.path.join("static", heatmap_filename)

    cv2.imwrite(heatmap_path, superimposed_img)

    return heatmap_path, position, round(area_percentage, 2), size_category


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join("static", file.filename)

            # Save image FIRST
            file.save(filepath)

            # Predict
            label, confidence = predict_image(filepath)

            # Generate Grad-CAM
            heatmap_path, position, area_percentage, size_category = generate_gradcam(filepath)

            # Only show position if stone detected
            if label != "Stone Detected":
                position = "Not Applicable"
                area_percentage = 0
                size_category = "Not Applicable"

            # 🔹 RISK LOGIC (ADD THIS)
            if label == "Stone Detected":
                if confidence > 90:
                    risk = "High Risk"
                elif confidence > 75:
                    risk = "Moderate Risk"
                else:
                    risk = "Low Risk"
            else:
                risk = "No Risk"

            # 🔹 CLINICAL NOTE
            if label == "Stone Detected":
                if confidence > 90:
                    warning = "High probability of renal calculi detected. Immediate urological evaluation is recommended."
                elif confidence > 75:
                    warning = "Moderate probability of kidney stone presence. Clinical correlation and further imaging may be required."
                else:
                    warning = "Low confidence detection. Recommend clinical verification and radiologist review."
            else:
                if confidence > 90:
                    warning = "No radiological evidence of renal calculi detected in the analyzed image."
                else:
                    warning = "No clear evidence of kidney stone. However, clinical symptoms should be correlated."                
                
            pdf_path = generate_pdf_report(
                                      label, confidence, risk, position,
                                        area_percentage, size_category, warning,
                                         filepath, heatmap_path
                                            )
            
            return render_template(
                "result.html",
                label=label,
                confidence=confidence,
                risk=risk,
                position=position,
                area_percentage=area_percentage,
                size_category=size_category,
                image_path=filepath,
                warning=warning,
                pdf_path=pdf_path,
                heatmap_path=heatmap_path
            )
            
            
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)