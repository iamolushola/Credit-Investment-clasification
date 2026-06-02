# Streamlit Dashboard Setup

This version lets you run the ML workflow locally on your PC with Streamlit instead of running the Jupyter notebook line by line.

## 1. Open PowerShell or CMD in the project folder

```powershell
cd C:\Users\user\Downloads\ML_Project\borrower_investor_ml_project
```

## 2. Activate your virtual environment

PowerShell:

```powershell
venv\Scripts\activate
```

If PowerShell blocks activation, use CMD:

```cmd
venv\Scripts\activate.bat
```

## 3. Install requirements

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Run Streamlit

```powershell
python -m streamlit run streamlit_app.py
```

Streamlit will open a browser page like:

```text
http://localhost:8501
```

## 5. How to use the dashboard

1. Upload your Excel or CSV file, or use the included file in `data/raw/`.
2. Confirm the sheet name is `REAL DATA`.
3. Select a model.
4. Click **Train / Refresh Model**.
5. Review:
   - dataset preview
   - class balance
   - confusion matrix
   - macro/micro F1
   - per-class precision/recall/F1
   - predictions table
6. Download the predictions CSV.

## 6. Important note

The current dataset does not contain verified labels, so the app generates pseudo-labels using the rule-based strategy in `src/data_processing.py`.

The model performance shows how well the model learned those pseudo-labels. For production use, validate the labeling rules with domain experts or add real labeled examples.
