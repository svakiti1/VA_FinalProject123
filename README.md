# 🏥 MIMIC-III Healthcare Analytics Dashboard

<p align="center">
  <img src="mimic3.png" alt="MIMIC-III Dashboard" width="400"/>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11" />
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/streamlit-%E2%89%A5--1.0-orange" alt="Streamlit" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" />
  </a>
</p>

---

## ✨ Overview

This **Streamlit** application delivers a dynamic and visually engaging interface for exploring the **MIMIC-III** dataset. Leverage ready-made charts, interactive filters, and data summaries to gain insights into patient demographics, hospital admissions, ICU stays, diagnoses, procedures, lab results, and medication patterns.

---

## 🚀 Features

| Section                  | Description                                                            |
|--------------------------|------------------------------------------------------------------------|
| 🗺️ **Overview & Demographics** | Age distribution, gender split, and ethnicity composition.            |
| 🏨 **Admissions Analysis**     | Admission trends, length of stay, and discharge outcomes.             |
| 🛏️ **ICU Stays**              | ICU admission sources, stay durations, and mortality rates.           |
| 🩺 **Diagnoses Explorer**      | Filterable ICD code lookup with top diagnoses frequency charts.       |
| 🔧 **Procedures Analysis**     | Common procedures overview with temporal trends.                      |
| 🔬 **Laboratory Results**      | Key lab metric distributions and time-series visualizations.          |
| 💊 **Medication Analysis**     | Prescription patterns, drug frequency, and dosage breakdowns.         |

---

## 📂 Directory Structure

```bash
.
├── app.py                   # Main Streamlit app script
├── data/                    # Cleaned MIMIC-III datasets in Parquet format
│   ├── ADMISSIONS_clean.parquet
│   ├── DIAGNOSES_ICD_clean.parquet
│   ├── D_ICD_DIAGNOSES_clean.parquet
│   ├── D_ICD_PROCEDURES_clean.parquet
│   ├── D_LABITEMS_clean.parquet
│   ├── ICUSTAYS_clean.parquet
│   ├── LABEVENTS_clean.parquet
│   ├── PATIENTS_clean.parquet
│   ├── PRESCRIPTIONS_clean.parquet
│   └── PROCEDURES_ICD_clean.parquet
├── mimic3.png               # Dashboard logo/image
├── requirements.txt         # Python dependencies
└── runtime.txt              # Runtime spec (Python 3.11)
```

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mimic3-dashboard.git
   cd mimic3-dashboard
   ```
2. **Create & activate a Python 3.11 environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\\Scripts\\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Load data**:
   - Place cleaned Parquet files into the `data/` directory.

---

## ▶️ Usage

Launch the dashboard:
```bash
streamlit run app.py
```
- Access at `http://localhost:8501` in your browser.
- Use the sidebar 🔍 navigation to explore each section.

---

## 🧹 Data Loading & Preprocessing

```python
with st.spinner("Loading MIMIC-III data..."):
    data_dict = load_data()
    data_dict = preprocess_data(data_dict)
```
- **load_data()**: Reads Parquet files into Pandas DataFrames.
- **preprocess_data()**: Cleans, merges, and computes features for visualization.

---

## ⚙️ Configuration

- **Page layout**:
  ```python
  st.set_page_config(
      layout="wide",
      page_title="MIMIC-III Dashboard",
      page_icon="🏥"
  )
  ```
- **Version**: v1.0

---
