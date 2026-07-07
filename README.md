# Adelaide AU: Attention Measurement Simulator 👁️

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-172B4D?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**🌟 Live Demo:** [https://adelaide-mvp-sjsx8wnrerjuycvx82whdp.streamlit.app/](https://adelaide-mvp-sjsx8wnrerjuycvx82whdp.streamlit.app/)

This project is a functional prototype of an **Attention Measurement and Optimization Platform**, inspired by Adelaide's industry-leading AU metric. It demonstrates an end-to-end data science workflow from synthetic data generation and predictive modeling to causal inference and interactive visualization.

## 🚀 Features

The application features an interactive Streamlit dashboard divided into three core components:

1. **Attention Predictor (Optimization Engine):** 
   - Allows users to act as media buyers/advertisers.
   - Users can input ad parameters (Format, Page Position, Device) and environmental factors (Page Clutter, Domain Quality, Scroll Velocity, Time in View).
   - Utilizes pre-trained **XGBoost** and **Random Forest** models to predict the expected Attention Score in real-time.
   - Includes a UI toggle to instantly compare predictions between the two algorithms.

2. **Insights & Exploratory Data Analysis (EDA):**
   - Visualizes the key drivers of attention across 20,000 simulated ad impressions.
   - Highlights the relationships between page clutter, ad formats, and resulting attention scores to tell a compelling, data-driven story.

3. **Causal Impact (Inverse Probability Weighting):**
   - Moves beyond correlation to prove causality.
   - Implements a simplified Causal Inference pipeline (using IPW) to demonstrate that high-attention ads *cause* a higher conversion rate, even when controlling for confounding variables like premium publisher domains.

---

## 🧠 Methodology

### Synthetic Data Generation (`model_pipeline.py`)
Since proprietary attention data is heavily guarded, this project generates a highly realistic synthetic dataset of 20,000 ad impressions. The data incorporates realistic relationships (e.g., highly cluttered pages severely penalize attention; native ads perform better than standard banners).

### Machine Learning
The predictive engine utilizes two powerful tree-based algorithms:
- **XGBoost Regressor**
- **Random Forest Regressor** (scikit-learn)

Both models are trained to predict the continuous `attention_score` target. The models and their corresponding feature maps are serialized using `joblib` into an `attention_model.pkl` artifact for rapid inference in the web app.

---

## 🛠️ Installation and Local Setup

To run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sarveshKharche/adelaide-mvp.git
   cd adelaide-mvp
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note for Mac users: XGBoost requires OpenMP. If you encounter issues during installation, run `brew install libomp` first).*

4. **(Optional) Re-generate the Data & Models:**
   The repository comes with pre-trained models. If you wish to regenerate the synthetic data and retrain the models from scratch, run:
   ```bash
   python model_pipeline.py
   ```

5. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

```
├── app.py                  # The main Streamlit dashboard application
├── model_pipeline.py       # Script for synthetic data generation and model training
├── attention_model.pkl     # Serialized dictionary containing the RF & XGBoost models
├── synthetic_ad_data.csv   # The generated dataset used for EDA in the app
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
