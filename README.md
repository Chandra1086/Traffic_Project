# 🚦 Automated Traffic Signal & Hand Gesture Detection

## 📌 Project Overview

This project implements an **AI-based computer vision prototype** capable of detecting **traffic signals and traffic police hand gestures in real time** using a **laptop webcam**.
The system is designed as a **low-cost, small-scale prototype** without integration into real vehicle control systems.

The project uses **deep learning and transfer learning** to classify traffic signals and hand gestures from live video input.

---

## 🎯 Objectives

* To develop an AI-based vision prototype using deep learning models for real-time detection of **traffic signals and hand gestures**.
* To evaluate and demonstrate the model’s performance on a **basic hardware setup (laptop + webcam)** using accuracy-based metrics.

---

## 🧠 Technologies Used

* **Python 3.10**
* **TensorFlow / Keras**
* **MobileNetV2 (Transfer Learning)**
* **OpenCV**
* **NumPy**
* **VS Code**
* **Webcam (Live Video Input)**

---

## 📂 Project Structure

```
Traffic_Project/
│
├── DataSet/
│   ├── 0-Green Light
│   ├── 1-Red Light
│   ├── 2-Yellow Light
│   ├── 3-lane left
│   ├── 4-lane right
│   ├── 5-left
│   ├── 6-left over
│   ├── 7-left turn
│   ├── 8-move straight
│   ├── 9-right
│   ├── 10-right over
│   ├── 11-right turn
│   └── 12-stop signal
│
├── train_model.py
├── live_predict.py
├── traffic_hand_signal_cnn.h5
├── venv/
└── README.md
```

---

## ⚙️ Model Description

* **Backbone:** MobileNetV2 (pre-trained on ImageNet)
* **Input Size:** 224 × 224
* **Classes:** 13 (Traffic lights + Hand gestures)
* **Training Approach:** Transfer learning with frozen base layers
* **Output:** Softmax-based multi-class classification

---

## ▶️ How to Run the Project (VS Code – Step by Step)

### 🔹 Step 1: Open VS Code

* Press **Windows key**
* Search **VS Code**
* Open it

---

### 🔹 Step 2: Open Project Folder

* Go to **File → Open Folder**
* Select:

  ```
  Desktop → Traffic_Project
  ```
* Click **Select Folder**

---

### 🔹 Step 3: Open Terminal

* Press **Ctrl + `** (backtick key below ESC)
  **OR**
* Menu → **Terminal → New Terminal**

You should see:

```
PS C:\Users\...\Traffic_Project>
```

---

### 🔹 Step 4: Activate Virtual Environment (IMPORTANT)

Run:

```powershell
.\venv\Scripts\activate
```

After activation, you must see:

```
(venv) PS C:\Users\...\Traffic_Project>
```

---

### 🔹 Step 5: (Optional) Train the Model Again

⚠️ Skip this step if `traffic_hand_signal_cnn.h5` already exists.

```powershell
python train_model.py
```

After training:

```
✅ Model saved as traffic_hand_signal_cnn.h5
```

---

### 🔹 Step 6: Run Live Detection (Main Step)

```powershell
python live_predict.py
```

---

## 🎥 Expected Output

* Webcam opens automatically
* A **large green ROI box** appears
* Predicted signal label and confidence shown, for example:

  ```
  Red Light (54%)
  Move Straight (63%)
  Stop Signal (71%)
  ```

---

## ⛔ Exit the Application

* Press **`q`** inside the webcam window
* Terminal output:

  ```
  Webcam stopped
  ```

---

## 🧪 Evaluation Metrics

* Training Accuracy
* Validation Accuracy
* Confidence Scores (Live Prediction)
* Real-time responsiveness

---

## ⚠️ Common Issues & Fixes

### ❌ Webcam not opening?

* Close Zoom / Teams / browser tabs using the camera

---

### ❌ `(venv)` not showing?

Run again:

```powershell
.\venv\Scripts\activate
```

---

### ❌ Model file not found?

Ensure:

```
traffic_hand_signal_cnn.h5
```

exists in the project root folder.

---

## 🚀 Future Enhancements

* Integrate **YOLOv8** for bounding-box-based detection
* Separate pipelines for **hand gestures and traffic lights**
* Add **audio alerts** (STOP / GO)
* Improve accuracy with larger datasets
* Convert project into a standalone application

---

## 👨‍💻 Developed By

**A.V. Chandrakanth Reddy**
B.Tech CSE – GITAM University, Bengaluru

---

## 📜 License

This project is developed for **academic and educational purposes only**.

---
