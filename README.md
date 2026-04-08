# 🌍 Global Development Categorization Dashboard

An end-to-end **unsupervised machine learning** project that clusters countries by socio-economic indicators and visualizes global development patterns through an interactive dashboard.

🔗 **Live Demo:** [global-development-jpa3hswfotyh3mxnkocgfg.streamlit.app](https://global-development-jpa3hswfotyh3mxnkocgfg.streamlit.app/)

---

## 📌 Project Overview

This project analyzes world development data across 22 socio-economic indicators — including GDP, health expenditure, CO2 emissions, internet usage, and population metrics — to automatically group countries into development categories using K-Means clustering.

The goal is to answer: **Can we objectively classify countries by development level using only data?**

---

## 🖼️ Dashboard Preview

| World Map | Country Analysis |
|-----------|-----------------|
| Choropleth map colored by development level | Radar chart comparing a country vs its cluster average |

---

## ⚙️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3 |
| ML | Scikit-Learn (KMeans, SimpleImputer, StandardScaler) |
| Data | Pandas, NumPy |
| Visualization | Plotly Express |
| App Framework | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📂 Project Structure

```
Global-Development/
│
├── app.py                          # Streamlit dashboard
├── train_model.py                  # Preprocessing pipeline & artifact generation
├── World_development_mesurement.xlsx  # Raw dataset
├── requirements.txt                # Dependencies
│
└── model/
    ├── imputer.pkl                 # Fitted SimpleImputer
    ├── scaler.pkl                  # Fitted StandardScaler
    └── features.pkl                # Feature column names
```

---

## 🧠 ML Pipeline

```
Raw Data
   │
   ▼
Special Character Removal  ──  ($, %, , in GDP / Tax Rate / Tourism cols)
   │
   ▼
Drop Irrelevant Columns  ──  (Ease of Business >50% missing, Number of Records)
   │
   ▼
Median Imputation  ──  (SimpleImputer — handles remaining nulls)
   │
   ▼
Log Transformation  ──  (np.log1p on 11 right-skewed columns)
   │
   ▼
Standard Scaling  ──  (StandardScaler — zero mean, unit variance)
   │
   ▼
K-Means Clustering  ──  (k=2 to 6, user-controlled via sidebar)
   │
   ▼
GDP-Based Label Assignment  ──  (Under-Developed → Developing → Developed ...)
```

---

## 📊 Features

- **Interactive World Map** — choropleth colored by development level
- **Adjustable Clusters** — sidebar slider lets you change k from 2 to 6 live
- **Country Drilldown** — zoom into any individual country on the map
- **Radar Chart** — compare a selected country's scaled indicators vs its cluster average
- **Distribution Plot** — box plot of any indicator split by development level
- **Single Country Analysis** — indicator-by-indicator comparison vs global average with status labels (Good / Average / Needs Improvement)
- **CSV Download** — download a full development report for any country

---

## 📈 Key Results (k=3)

| Cluster | Label | Characteristics |
|---------|-------|-----------------|
| 0 | Under-Developed | Low GDP, high birth/infant mortality rate, low internet usage |
| 1 | Developing | Mid-range GDP, growing mobile/internet penetration |
| 2 | Developed | High GDP, high health expenditure, low mortality rates |

Silhouette Score at k=3: **~0.28** (meaningful separation given 22-dimensional data)

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/JAGGU-528/Global-Development.git
cd Global-Development

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate model artifacts (only needed once)
python train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

---

## 📋 Dataset

**World Development Measurement** — contains 2,700+ rows across 195+ countries with 24 original features including:

- Birth Rate, Infant Mortality Rate, Life Expectancy (Male/Female)
- GDP, Health Exp/Capita, Health Exp % GDP
- CO2 Emissions, Energy Usage
- Internet Usage, Mobile Phone Usage
- Tourism Inbound/Outbound
- Population (Total, Urban, Age groups)
- Business Tax Rate, Days/Hours to Start Business, Lending Interest

---

## 💡 What I Learned

- How to build a **complete ML pipeline** from raw messy data to a deployed web app
- Why **preprocessing order matters** — impute before log transform, not after
- How **pickle version pinning** is critical for reproducible deployments
- How to use `@st.cache_data` and `@st.cache_resource` correctly in Streamlit
- Practical difference between **K-Means, DBSCAN, Hierarchical, and GMM** clustering

---

## 👤 Author

**Jagadeesh** — Electronics & Communication Engineering Graduate  
Targeting Data Science / ML Engineer roles  

[![GitHub](https://img.shields.io/badge/GitHub-JAGGU--528-black?logo=github)](https://github.com/JAGGU-528)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit)](https://global-development-jpa3hswfotyh3mxnkocgfg.streamlit.app/)

---

> *Built as part of a structured Data Science learning roadmap. All preprocessing, model training, deployment debugging, and dashboard development done independently.*
