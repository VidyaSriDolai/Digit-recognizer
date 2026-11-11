# Handwritten Digit Recognizer (CNN + Streamlit)

Tabs: Draw | Upload Image | Model Performance

## Run locally
pip install -r requirements.txt
python train_model.py        # ~2-3 min; creates digit_model.keras, training_curves.png,
                             # confusion_matrix.png, metrics.json
streamlit run app.py

## Google Colab (for training only)
Upload train_model.py, run `!python train_model.py`, then download all 4 generated files
and place them next to app.py.

## Deploy (Streamlit Community Cloud)
Push app.py, requirements.txt, digit_model.keras, metrics.json, training_curves.png and
confusion_matrix.png to GitHub. Deploy with app.py as entry point (Python 3.11).
