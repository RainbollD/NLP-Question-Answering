from transformers import pipeline
import torch
from get_configs import Paths


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    return pipeline("question-answering", model=Paths().get_model_rus_qa_path())


def predict_ru_qa(question, context):
    """
    Create function-model and add answers in the loop
    :return: answer
    """
    model_qa = load_model()

    result = model_qa(question=question, context=context)

    return result

