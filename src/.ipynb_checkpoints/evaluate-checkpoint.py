from data_loader import load_data
from model import train_model

def test_model_training_and_accuracy():
    df = load_data("dataset/iris.csv")
    model, acc = train_model(df)
    assert model is not None
    assert hasattr(model, "predict")
    assert acc >= 0.90  
