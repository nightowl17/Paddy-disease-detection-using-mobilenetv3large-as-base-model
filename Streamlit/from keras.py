from keras.models import load_model
from pathlib import Path

model_path = Path(__file__).parent / "Mobilenetv3large_paddy_disease_detection_architecture_3_fine_tuned.keras"
model = load_model(model_path)
