# grp39-AI4SE-Wk-2-Assignment
Group 39 Assignment 2 (AI for Software Engineering)

# 🛡️ MalariaShield: Predicting Future Malaria Cases for SDG 3

**Goal:** To build a machine learning model that predicts future malaria cases based on historical incidence, reported, and estimated data.

**Why:** This project supports Sustainable Development Goal 3 — "Good Health and Well-being" — by enabling health authorities to plan malaria prevention and response.

## 📁 Data Sources:
- Estimated malaria cases and deaths
- Reported malaria data
- Incidence per 1000 population at risk

## ✅ Tasks:
1. Merge and clean the datasets.

---

## 🧹 Data Cleaning & Merging

The original datasets were:
- `estimated_numbers.csv`
- `reported_numbers.csv`
- `incidence_per_1000_pop_at_risk.csv`

### Steps Performed:
- Merged datasets on `Country`, `Year`, and `WHO Region`
- Filtered for only African countries (WHO Region: Africa)
- Filled missing values:
  - Replaced missing min/max values with the median
  - Filled missing reported values with `0` (assumed underreporting)
- Final cleaned dataset saved as: `malaria_africa_cleaned.csv`

---

## 📊 Streamlit Dashboard: MalariaShield

We built a lightweight web dashboard using [Streamlit](https://streamlit.io) to visualize malaria trends and support data-driven decision-making.

### Features:
- **Filter by country and year range**
- **Interactive line and bar charts** for:
  - Estimated vs Reported Malaria Cases
  - Estimated vs Reported Deaths
- **Future predictions** section placeholder for ML model integration

### 🔗 Try it on Streamlit Cloud
> [Live App Link](https://grp39-malaria-pred.streamlit.app/)

---

## 🛠️ Files in This Repo

| File                          | Description |
|-------------------------------|-------------|
| `estimated_numbers.csv`       | Raw dataset from WHO |
| `reported_numbers.csv`        | Raw reported malaria data |
| `incidence_per_1000_pop_at_risk.csv` | Raw incidence data |
| `Malaria Prediction.ipynb`    | Notebook for merging and cleaning data |
| `malaria_africa_cleaned.csv`  | Final cleaned dataset used in dashboard |
| `app.py`                      | Streamlit app script |
| `README.md`                   | Project documentation (this file) |

---

## 📌 Future Work

- Integrate a trained machine learning model to predict future malaria cases
- Add data upload functionality for real-time health data integration
- Include maps and more advanced visual analytics

---

## 👨‍💻 Built By:
**Group 39** — AI for Software Engineering, Week 2  
Powered by [PLP Africa](https://plpacademy.net)  
