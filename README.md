# Skin Image Colour Classifier

A Streamlit demo that loads a trained scikit-learn Random Forest model and predicts one of two **colour-based proxy categories** from an uploaded image.

## Important model limitation

Despite the original filenames `CNN.py` and `CNN.ipynb`, the uploaded training code creates a `RandomForestClassifier`, not a Convolutional Neural Network.

The training notebook creates labels from average RGB intensity:
- `Darker/Pigmented`
- `Lighter`

These are proxy labels, not medically validated skin-lesion diagnoses. This app does **not** detect skin cancer, distinguish benign from malignant lesions, or provide medical advice. Do not use its output for medical decisions.

## Project files

```text
skin_lesion_streamlit_app/
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally

Use Python 3.11 for the pinned dependencies.

1. Open a terminal in this project folder.
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Upload to GitHub

Create a new **public** or **private** repository on GitHub. Then, from this folder, run:

```bash
git init
git add app.py model.pkl requirements.txt README.md .gitignore
git commit -m "Add Streamlit image classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with your own GitHub details.

Alternatively, create the repository on GitHub and upload the five files through **Add file → Upload files**.

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ using GitHub.
2. Select **Create app**.
3. Choose the repository, branch `main`, and main file path `app.py`.
4. Select **Deploy**.
5. Wait for the app to install the packages in `requirements.txt` and start.

If deployment fails, open the app's **Manage app → Logs** and check the error message.

## Model compatibility

The supplied pickle emitted a scikit-learn version warning when inspected: it was created with scikit-learn 1.6.1, while the inspection environment had 1.8.0. This project pins `scikit-learn==1.6.1` to match the model's recorded version as closely as possible.

Only load pickle files that you trust. Pickle files can execute code when loaded.
