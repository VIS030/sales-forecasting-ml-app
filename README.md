# AI Sales Prediction System

A modern end-to-end **Sales Prediction Web Application** built with **Flask**, **SQLite**, and **Machine Learning**, designed with a professional SaaS-style dashboard UI.

This project uses a large e-commerce dataset to:
- clean and store transactional data,
- generate analytics and EDA charts,
- train multiple ML models,
- and provide real-time sales prediction through a web interface and API.

---

## Features

### Core Data + ML
- Loads and cleans `ecommerce_dataset.csv` (100K+ rows)
- Removes duplicate columns and handles missing values
- Converts date columns (`PurchaseDate`, `SignUpDate`) to datetime
- Stores cleaned data in SQLite (`sales.db`, table: `sales_data`)
- Trains and compares:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor (if available)
- Selects best model by metrics (`R2`, `MAE`) and saves as `model.pkl`

### Dashboard + Analytics
- KPI cards:
  - Total Revenue
  - Total Orders
  - Unique Users
  - Avg Rating
- Interactive Chart.js visualizations:
  - Sales Over Time (line)
  - Top Products (bar)
  - Category Distribution (pie)
  - Discount vs Sales (scatter)
- Filters: Date range, Category, Country, Gender

### Authentication (Session-Based)
- Signup / Login / Logout
- SQLite `users` table with hashed passwords
- Protected routes:
  - `/dashboard`
  - `/predict`
  - `/eda`
  - `/api/predict`
- Flash messages for success and error states

### Additional Pages
- `/eda` page with generated charts and insights
- `/about` page with detailed project explanation

---

## Tech Stack

- **Backend:** Flask, Python
- **Database:** SQLite
- **ML/Data:** Pandas, Scikit-learn, XGBoost, Seaborn, Matplotlib
- **Frontend:** Bootstrap 5, Chart.js, Font Awesome, custom CSS

---

## Project Structure

```text
project/
├── app.py
├── ecommerce_dataset.csv
├── model.pkl
├── sales.db
├── README.md
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── login.html
│   ├── signup.html
│   ├── eda.html
│   └── about.html
└── static/
    ├── css/
    │   └── style.css
    ├── js/
    │   └── script.js
    └── eda/
        ├── sales_trend.png
        ├── category_distribution.png
        ├── age_distribution.png
        ├── gender_distribution.png
        └── discount_vs_sales.png
```

---

## Setup Instructions

## 1) Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

## 2) Create and activate virtual environment

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Install dependencies

```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn joblib xgboost
```

## 4) Run the app

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

---

## First Run Behavior

On first launch, the app automatically:
- loads and cleans the dataset,
- creates/updates `sales.db`,
- generates EDA chart images in `static/eda/`,
- trains models and saves `model.pkl`.

This may take some time depending on your machine.

---

## Routes

### Public
- `/` - Home
- `/login` - Login
- `/signup` - Signup
- `/about` - About
- `/logout` - Logout

### Protected (Login Required)
- `/dashboard` - Analytics dashboard
- `/predict` - Prediction page
- `/eda` - EDA charts and insights
- `/api/predict` - Prediction API (POST)

---

## API Usage

### Endpoint

`POST /api/predict`

### JSON Request Body

```json
{
  "category": "Books",
  "country": "USA",
  "price": 120.5,
  "quantity": 3,
  "discount_rate": 0.1
}
```

### JSON Response

```json
{
  "predicted_sales": 359.82,
  "confidence_score": 99.0,
  "model": "Random Forest",
  "evaluated_r2": 1.0,
  "evaluated_mae": 0.0687
}
```

---

## Screenshots (Recommended for GitHub)

Add screenshots in a folder like `docs/screenshots/` and update links below:

- Home page
- Dashboard page
- Predict page
- EDA page
- Login/Signup page

Example markdown:

```md
![Dashboard](docs/screenshots/dashboard.png)
```

---

## Security Notes

- Passwords are hashed before storage.
- Session auth is enabled for protected pages.
- For production, set a strong secret key:

```bash
FLASK_SECRET_KEY=your-strong-secret
```

---

## Future Improvements

- Role-based access (admin/analyst)
- Export analytics reports (CSV/PDF)
- Docker support for easy deployment
- CI/CD pipeline and automated tests

---

## License

This project is open-source and available under the **MIT License**.
