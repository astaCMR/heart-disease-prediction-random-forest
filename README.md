# ❤️ Heart Disease Prediction Using Random Forest

## 📌 Project Overview
Heart disease is one of the major health problems worldwide. Early prediction can help in preventing serious complications.  
This project predicts whether a person has heart disease using a **Random Forest Machine Learning model** and provides results through a **Flask-based web application**.

The system integrates **Machine Learning + Web Development** to create an end-to-end predictive application.

---

## 🎯 Objectives
- To predict the risk of heart disease using medical parameters  
- To build a user-friendly web interface for prediction  
- To integrate a machine learning model with Flask  
- To demonstrate a complete ML mini project  

---

## 🧠 Machine Learning Model
- **Algorithm:** Random Forest Classifier  
- **Why Random Forest?**
  - High accuracy
  - Handles non-linear data
  - Reduces overfitting
  - Suitable for medical datasets  

---

## 📊 Dataset Information
- **Dataset Name:** Heart Disease Dataset  
- **Total Features:** 13  
- **Target Column:** `target`  
  - `0` → No Heart Disease  
  - `1` → Heart Disease  

### Input Features:
1. Age  
2. Sex  
3. Chest Pain Type (cp)  
4. Resting Blood Pressure (trestbps)  
5. Cholesterol (chol)  
6. Fasting Blood Sugar (fbs)  
7. Resting ECG (restecg)  
8. Maximum Heart Rate (thalach)  
9. Exercise Induced Angina (exang)  
10. Oldpeak  
11. Slope  
12. Number of Major Vessels (ca)  
13. Thalassemia (thal)  

---

## 🖥️ Technologies Used

| Category | Tools |
|--------|-------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Version Control | Git, GitHub |

---

## 🏗️ Project Structure

heart-disease-prediction/
│
├── app.py
├── heart.csv
├── requirements.txt
├── README.md
│
├── templates/
│ └── index.html
│
└── static/
└── style.css


---

## ⚙️ Working of the System
1. User enters medical details in the web form  
2. Flask collects the input data  
3. The Random Forest model processes the data  
4. Prediction result is generated  
5. Result is displayed on the web page  

---

## ▶️ How to Run the Project Locally

This guide will help you run the Heart Disease Prediction application on your local machine. You only need **Python** installed to get started!

### 📋 Prerequisites
- **Python 3.7+** installed on your system
- A command line/terminal application
- A web browser (Chrome, Firefox, Edge, Safari, etc.)

#### Verify Python Installation
Open your terminal/command prompt and run:
```bash
python --version
```
You should see something like `Python 3.x.x`. If not, [download Python](https://www.python.org/downloads/).

---

### 🚀 Installation & Setup Steps

#### Step 1: Clone or Download the Repository
If you have Git installed:
```bash
git clone https://github.com/astaCMR/heart-disease-prediction-random-forest.git
cd heart-disease-prediction-random-forest
```

Or download the ZIP file from GitHub and extract it.

#### Step 2: Navigate to Project Directory
```bash
cd Heart-Disaese-Prediction-Using-Random-Forest
```

#### Step 3: Create a Virtual Environment (Recommended)
A virtual environment isolates project dependencies from your system Python.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, your terminal should show `(venv)` at the beginning of the line.

#### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install all required packages:
- **Flask** - Web framework for the UI
- **NumPy** - Numerical computing
- **Pandas** - Data processing
- **Scikit-learn** - Machine Learning library

The installation process may take 1-2 minutes depending on your internet speed.

#### Step 5: Run the Flask Application
```bash
python run_local.py
```

You should see output similar to:
```
 * Serving Flask app 'api.index'
 * Debug mode: on
WARNING: This is a development server. Do not use it in production deployment.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

#### Step 6: Open in Your Browser
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

🎉 **The application is now running locally!**

---

### 📝 Using the Application

1. **Fill in Medical Information**: Enter the 13 medical parameters in the form
   - Hover over the **ℹ️ info icons** next to each field to see what each parameter means
   - All fields are required

2. **Submit the Form**: Click the "Predict" button

3. **View Results**: The prediction result will appear:
   - ✅ **No Heart Disease** - Low risk of heart disease
   - ⚠️ **Heart Disease Detected** - Higher risk of heart disease

---

### 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `Command 'python' not found` | Python is not installed or not in PATH. Download from [python.org](https://www.python.org/downloads/) |
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` again |
| `Address already in use` | Port 5000 is busy. Change port in `run_local.py` or restart your system |
| `PermissionError` when accessing `/api` | Run terminal as administrator (Windows) or use `sudo` (macOS/Linux) |
| Browser shows "Connection refused" | Ensure Flask is running and port 5000 is accessible |

---

### ⏹️ Stopping the Application

Press `CTRL+C` in your terminal to stop the Flask server.

To deactivate the virtual environment:
```bash
deactivate
```

---

### 🔄 Running Again Later

After closing the application, to run it again:

1. Open terminal in the project directory
2. Activate virtual environment:
   - **Windows**: `venv\Scripts\activate`
   - **macOS/Linux**: `source venv/bin/activate`
3. Run: `python run_local.py`
4. Open browser to `http://127.0.0.1:5000`

---

### 📦 Project Files Explained

| File/Folder | Purpose |
|-------------|---------|
| `run_local.py` | Entry point - runs the Flask application |
| `api/index.py` | Backend logic - Flask routes and ML model |
| `templates/index.html` | Frontend - HTML form and UI |
| `static/style.css` | Styling - CSS for the web interface |
| `heart.csv` | Dataset - 303 patient records used for training |
| `requirement.txt` | List of Python dependencies |

## 📈 Model Performance

Training Accuracy: ~100%

Testing Accuracy: ~66%

Note: Slight overfitting is expected due to small dataset size and is acceptable for academic projects.

## ✅ Output

No Heart Disease
<img width="1600" height="899" alt="image" src="https://github.com/user-attachments/assets/7a8d4663-0d57-4cdd-8cc2-d3a9ee157a6e" />

Heart Disease Detected
<img width="1600" height="899" alt="image" src="https://github.com/user-attachments/assets/9836e8d5-acc9-42eb-9dee-c50c4afa3d62" />

The result is displayed immediately after submitting the form.


## ✅ Result
The Random Forest model effectively predicts heart disease using patient health parameters.
