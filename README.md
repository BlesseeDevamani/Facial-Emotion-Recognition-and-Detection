# 😊 Face Emotion Recognition using Machine Learning

## 📌 Overview
This project implements a Face Emotion Recognition System using Machine Learning and Deep Learning techniques.  
The model detects human faces from images or real-time webcam input and classifies emotions into:

- Happy
- Sad
- Angry
- Surprise
- Fear
- Neutral
- Disgust

The system is built using Computer Vision and Convolutional Neural Networks (CNNs).

---

## 🎯 Features
- Face detection using OpenCV
- Emotion classification using CNN
- Real-time webcam emotion recognition
- Image-based emotion prediction
- Trained on FER-2013 dataset

---

## 🛠️ Tech Stack
- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 📂 Project Structure

```
Face-Emotion-Recognition/
│
├── dataset/
│   └── fer2013.csv
│
├── model/
│   └── emotion_model.h5
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── realtime.py
│   └── preprocess.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset
The project uses the FER-2013 dataset:
- 48x48 grayscale facial images
- 7 emotion categories
- Training and testing split included

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Face-Emotion-Recognition.git
cd Face-Emotion-Recognition
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### ▶ Train the Model
```bash
python src/train.py
```

### ▶ Predict Emotion from Image
```bash
python src/predict.py --image path_to_image.jpg
```

### ▶ Real-Time Emotion Detection
```bash
python src/realtime.py
```

---

## 🧠 Model Architecture
The CNN model consists of:
- Convolutional Layers
- ReLU Activation
- MaxPooling Layers
- Dropout Layers
- Fully Connected Dense Layers
- Softmax Output Layer

**Loss Function:** Categorical Crossentropy  
**Optimizer:** Adam  
**Metric:** Accuracy  

---

## 📈 Results
- Achieved good validation accuracy.
- Real-time emotion detection works successfully.
- Performs best under proper lighting conditions.

---

## 🔮 Future Improvements
- Improve accuracy using Transfer Learning (VGG16, ResNet)
- Deploy as a Web App using Flask or Streamlit
- Add support for multiple face detection
- Deploy on cloud platforms

---

## 🤝 Contributing
Contributions are welcome!  
Fork the repository and submit a pull request.

---

## 📜 License
This project is licensed under the MIT License.

---

## 👩‍💻 Author
Yadiki Blessee Devamani  
Btech-Computer Science and Engineering  
blesseedevamani751@gmail.com
# Face_Emotion_Recognition_Machine_Learning
Face Emotion Recognition using Machine Learning Python

Watch Tutorial :- https://www.youtube.com/watch?v=aoCIoumbWQY
