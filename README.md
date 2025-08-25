# 🩺 Pneumonia Detection App  

A deep learning-powered web app to detect **pneumonia from chest X-rays** using **PyTorch** and **Grad-CAM** for explainability.  
Deployed seamlessly on **Hugging Face Spaces** with Gradio.  

---

## 🚀 Live Demo  
👉 [Try the App on Hugging Face](https://huggingface.co/spaces/YOUR_USERNAME/pneumonia-app)  

---

## 📸 Screenshots  

| Upload Screen | Prediction Result | Grad-CAM Visualization |
|---------------|------------------|-------------------------|
| ![Upload](screenshots\demo.PNG) | ![Prediction](screenshots\prediction.PNG) | ![Grad-CAM](screenshots\gradcam.PNG) |

---

## ⚙️ Tech Stack  

- **Frontend:** Gradio  
- **Backend:** PyTorch, Torchvision  
- **Deployment:** Hugging Face Spaces  
- **Visualization:** Grad-CAM, Matplotlib  

---

## 🛠️ Installation  

Clone the repo and set up locally:  

```bash
git clone https://github.com/TANMaYtO/pneumonia-detection-app
cd pneumonia-detection-app
pip install -r requirements.txt
gradio app.py
    
---

## 📌 Usage  

1. Upload a **chest X-ray image**  
2. The app predicts:  
   - ✅ Normal  
   - ⚠️ Pneumonia  
3. A heatmap is generated via **Grad-CAM** to highlight critical regions.  

---

## 🚀 Features  

- 🧠 Deep learning CNN trained on chest X-ray dataset  
- 🔥 Grad-CAM for explainability  
- 📈 Data augmentation + optimizer tuning for robustness  
- ☁️ One-click deployment on Hugging Face  

---

## 💡 Lessons Learned  

- Built a full ML workflow: **training → evaluation → deployment**  
- Gained experience in **model interpretability** for healthcare AI  
- Learned deployment best practices with **Gradio + Hugging Face**  

---

## 📜 License  

MIT License © 2025 [Tanmay Tomar]  
