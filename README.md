# House Price Predictor 

A Streamlit web app that loads a Random Forest mddel trained for the
House Price India dataset and lets you enter house features to get a live
price prediction.

## Folder Contents

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application |
| `best_house_price_model.pkl` | The trained regression model (Random Forest with 100 trees and 15 depth limit |
| `house_price_scaler.pkl` | The fitted StandardScaler used during training (only actually used if the best model was Linear Regression) |
| `house_price_feature_columns.pkl` | The exact list/order of feature columns the model expects |
| `house_price_needs_scaling.pkl` | A flag (True/False) telling the app whether to scale inputs before prediction |
| `requirements.txt` | Python package dependencies |

All four `.pkl` files must be downloaded from the relatedColab notebook and
placed in this same folder, the app will not run without them.

## 1. Prerequisites

- Python 3.9+ installed (check with `python --version`)
- All 5 files above present in this folder

## 2. Create a Virtual Environment

Open a terminal in this folder (in VS Code: `Terminal → New Terminal`) and run:

```bash
python -m venv venv
```

## 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt once
it's active.

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs Streamlit, pandas, numpy, scikit-learn, and joblib at the
pinned versions listed in `requirements.txt`.

## 5. Run the App

```bash
streamlit run app.py
```

Your default browser should open automatically to:

```
http://localhost:8501
```

If it doesn't open automatically, copy that URL into your browser manually.

## 6. Using the App

1. Adjust the house feature inputs in the left sidebar (bedrooms, bathrooms,
   living area, lot area, condition, grade, year built, location, etc.).
2. Click **Predict Price**.
3. The estimated price will be displayed, along with the exact input row
   that was fed into the model.

You can expand **"Show exact feature columns expected by the model"** in the
sidebar to confirm the input fields match what the model was actually
trained on — if your dataset's column names differ slightly, adjust the
sidebar widgets in `app.py` accordingly.

## 7. Stopping the App

Go back to the terminal running Streamlit and press:

```
Ctrl + C
```

## 8. Running It Again Later

You do not need to recreate the virtual environment each time. Just:

```bash
source venv/bin/activate      # or venv\Scripts\activate on Windows
streamlit run app.py
```

## Troubleshooting

- **"Missing model artifact file(s)" error on launch** → one or more of the
  4 `.pkl` files is missing or misnamed. Confirm all four are in the same
  folder as `app.py` and match the exact filenames listed above.
- **`ModuleNotFoundError`** → the virtual environment isn't activated, or
  `pip install -r requirements.txt` wasn't run inside it. Re-check step 3
  and 4.
- **Port already in use** → run `streamlit run app.py --server.port 8502`
  to use a different port.