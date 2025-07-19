
# 💳 Credit Card Fault Detection

This project is a machine learning-powered web application that predicts whether a credit card transaction or user profile is **Good** or **Faulty (Bad)** based on various financial and demographic features. It is built using **Flask**, **Scikit-learn**, **XGBoost**, and deployed on **Render**.

🔗 **Live App:** [https://credit-default-classifier.onrender.com](https://credit-default-classifier.onrender.com)

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Dataset Description](#dataset-description)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Deployment](#deployment)
- [Author](#author)

---

## 📖 About the Project

This application helps financial institutions assess the **creditworthiness of customers** by predicting whether they are likely to **default** on their credit card payments.

It uses features such as:
- Credit limit
- Gender
- Education
- Marital status
- Payment history
- Bill amount
- Previous payments  
...and more.

---

## 📊 Dataset Description

The dataset includes the following important columns:

| Feature Name   | Description                                  |
|----------------|----------------------------------------------|
| LIMIT_BAL      | Amount of credit limit provided              |
| SEX            | Gender (1=Male, 2=Female)                    |
| EDUCATION      | Education level (1=graduate, 2=university...)|
| MARRIAGE       |  1 = Married,2 = Single, 3 = Others|    
    
                                
| AGE            | Age of the client                            |
| PAY_0 to PAY_6 | Payment delay in months

-2 = No consumption

-1 = Duly paid

0 = Paid minimum due

1 = 1 month delay

2 = 2 months delay

...

9 = ≥9 months delay"
| BILL_AMT1-6    | Previous bill statement amounts              |
| PAY_AMT1-6     | Amount paid in previous months               |
| default.payment.next.month | Target: 1=Default, 0=Not Default|

---

## ⚙️ Tech Stack

| Category       | Tools/Frameworks                             |
|----------------|----------------------------------------------|
| Language       | Python                                       |
| ML Libraries   | Scikit-learn, XGBoost                        |
| Web Framework  | Flask                                        |
| Frontend       | HTML, CSS (Jinja2 Templates)                 |
| Deployment     | Render                                       |
| Others         | Pandas, Matplotlib, Seaborn                  |

---

## 🗂 Project Structure

```bash
Credit-Card-Fault-Detection/
│
├── static/                
├── templates/
│   ├── index.html          # Landing page
│   └── home.html           # Result prediction form
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── utils.py
│
├── app.py                  # Flask application
├── requirements.txt        # Required libraries
├── setup.py                # For packaging
├── README.md               # This file
└── Procfile                # For deployment (gunicorn)
```

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Pragya26-techie/Detection_Model.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## ☁️ Deployment (Render)

- Created `requirements.txt` and `Procfile`
- Used gunicorn:  
  ```
  web: gunicorn app:app
  ```
- Connected GitHub repo to Render and deployed as a **Web Service**

---


---

## 👩‍💻 Author

**Pragya (Fresher | Python/ML Developer)**  


---

## ⭐️ Show your support

If you found this project helpful, give it a ⭐ on GitHub and consider sharing it with others!

---
