# BengaluruPrice 🏠
### ML-Powered House Price Predictor for Bengaluru

A full-stack machine learning web application that predicts residential property prices in Bengaluru — giving homebuyers a data-driven benchmark to negotiate fairly against broker-inflated quotes.

---

## The Problem

In India, buying a home means trusting a broker whose income depends on inflating the price. Most homebuyers have no way to verify if what they're paying is fair. **BengaluruPrice** solves this by providing an instant, transparent price estimate based on real market data.

---

## Live Demo

> Run locally — see setup instructions below.

---

## Features

- Predicts house prices based on locality, area, BHK, bathrooms, and balconies
- Covers **144 localities** across Bengaluru
- Trained on **9,243 real transactions**
- Clean two-page UI — landing page + predictor
- REST API endpoint for predictions

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Random Forest (scikit-learn) |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Data | Bengaluru House Price Dataset (Kaggle) |

---

## Model Performance

| Metric | Value |
|---|---|
| R² Score | 0.826 |
| MAE | ₹18.75 Lakhs |
| RMSE | ₹49.2 Lakhs |

> **Note:** High RMSE is a known limitation of predicting luxury properties (₹300L+) with limited training data. MAE of ₹18.75L is the more representative metric for typical properties.

---

## Project Structure

```
House_price_predictor/
├── dataset/
│   └── bengaluru_house_prices.csv
├── model/
│   ├── house_price_model.ipynb
│   ├── house_price_model.pkl
│   └── columns.json
├── templates/
│   ├── home.html
│   └── index.html
├── static/
├── app.py
├── requirements.txt
└── README.md
```

---

## Setup & Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/jay-joshi-web/house-price-predictor.git
cd house-price-predictor
```
> Download the dataset from Kaggle: "Bangluru house dataset" by Sanjay Chauhan
> Run all cells in model/house_price_model.ipynb to regenerate the pkl file
> Then run app.py

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python app.py
```

**4. Open in browser**
```
http://localhost:5000
```

---

## ML Pipeline

1. **Data Cleaning** — removed nulls, extracted BHK from size column, handled sqft ranges like "2100-2850"
2. **Outlier Removal** — removed statistical outliers per locality using mean ± 1 std deviation on price/sqft
3. **Feature Engineering** — created price_per_sqft for outlier detection, grouped 1,305 locations to 145
4. **Encoding** — One Hot Encoding on location column (144 dummy columns, drop_first=True)
5. **Log Transform** — applied np.log() on price to handle right-skewed distribution and luxury outliers
6. **Model Selection** — started with Linear Regression (R² 0.779, MAE ₹24L), upgraded to Random Forest (R² 0.826, MAE ₹18.75L)
7. **Deployment** — Flask REST API with Jinja2 HTML templates

---

## Challenges Faced

**High Cardinality in Location Feature**
The dataset had 1,305 unique locations. Direct OHE would create 1,304 columns on ~9,000 rows. Solved by grouping locations with fewer than 10 listings as "other", reducing to 145 locations.

**Luxury Property Outliers**
Linear Regression produced RMSE of ₹130L due to inability to model non-linear price behaviour in luxury properties. Switching to Random Forest reduced RMSE by 62% to ₹49L.

**Log Scale Evaluation Bug**
Model trained on log(price) — evaluation metrics must use np.exp() to convert predictions back to Lakhs. Initially computed MAE on log scale giving misleadingly small values.

---

## Future Scope

- Integrate live data via official real estate APIs
- Add city selector (Mumbai, Pune, Hyderabad)
- Use XGBoost for further accuracy improvement
- Deploy on Render / Railway for public access
- Add price trend visualization per locality

---

## Author

**Jay Joshi** — B.Tech AI/ML, 2nd Year

- LinkedIn: [jay-joshi-277734376](https://www.linkedin.com/in/jay-joshi-277734376)
- GitHub: [jay-joshi-web](https://github.com/jay-joshi-web)
- Email: jay.hamirott@gmail.com

---

*Built to give homebuyers the information they deserve.*