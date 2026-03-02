# kidney-stone-detection

🩺 Kidney Stone AI Diagnostic System

An AI-powered medical imaging system that detects kidney stones from CT scan images using a Convolutional Neural Network (CNN) with Grad-CAM explainability and generates a professional clinical PDF report.

🔬 Project Overview

This project implements a deep learning–based diagnostic assistant capable of:

Detecting kidney stones from CT scan images

Providing model confidence score

Estimating stone size and affected position

Generating Grad-CAM heatmap for model interpretability

Producing a professional clinical PDF diagnostic report

The system is built using TensorFlow/Keras for deep learning and Flask for deployment as a web-based medical AI application.

🧠 Model Architecture

The model uses a custom CNN architecture:

Conv2D (32 filters)

MaxPooling

Conv2D (64 filters)

MaxPooling

Conv2D (128 filters) ← used for Grad-CAM

MaxPooling

Dense Layer (128 units)

Dropout

Output Layer (Sigmoid activation)

Loss Function:
Binary Crossentropy

Optimizer:
Adam

🔥 Explainability – Grad-CAM

To ensure model transparency, Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented.

Grad-CAM:

Highlights regions influencing prediction

Helps validate model focus area

Improves clinical trust and interpretability

🏥 Diagnostic Output

For each CT image, the system provides:

Prediction (Stone / No Stone)

Confidence Score (%)

Risk Level (Low / Moderate / High)

Stone Position (Left / Right / Central / Not Applicable)

Estimated Stone Area (%)

Estimated Size (Small / Medium / Large)

Clinical Interpretation Note

Downloadable Professional PDF Report

📄 PDF Clinical Report Includes

Patient ID (Auto-generated)

Timestamp

Diagnostic Summary Table

Risk color coding

CT image

Grad-CAM heatmap

Clinical interpretation

Medical disclaimer

Designed to resemble a structured diagnostic document.

💻 Tech Stack

Backend:

Python

Flask

Deep Learning:

TensorFlow / Keras

NumPy

OpenCV

Visualization:

Matplotlib (colormap)

Grad-CAM

PDF Generation:

ReportLab

Version Control:

Git & GitHub


kidney-stone-detection/
│
├── app/
│   ├── static/
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── app.py
│
├── src/
│   ├── train.py
│   └── evaluate.py
│
├── models/
├── data/
├── README.md
└── .gitignore


🚀 How To Run Locally

1️⃣ Clone Repository

git clone https://github.com/sanku2025/kidney-stone-detection.git
cd kidney-stone-detection

2️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate  # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Train Model (Optional)

cd src
python train.py

5️⃣ Run Application
cd app
python app.py

Open browser:

http://127.0.0.1:5000

⚠️ Clinical Disclaimer

This system is an AI-assisted diagnostic tool intended for academic and research purposes.
It does not replace professional medical evaluation, radiological expertise, or clinical judgment.

📈 Future Improvements

Multi-class classification (stone type detection)

3D CT volume support

Better localization using segmentation models

Model performance metrics dashboard

Cloud deployment (AWS / GCP)

Docker containerization


👨‍💻 Author

B.Tech AI/ML Student
Deep Learning & Medical Imaging Enthusiast
