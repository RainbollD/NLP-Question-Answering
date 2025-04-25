from transformers import pipeline
import torch
from get_configs import Paths


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_model():
    return pipeline("question-answering", model=Paths().get_model_qa_path())


def predict_qa(data_questions_answers):
    """
    :type data_questions_answers: list
    :param data_questions_answers:
    :return:
    """
    model_qa = create_model()
    predictions = []
    for data_question_answer in data_questions_answers:
        info_model = model_qa(**data_question_answer)

        predictions.append(info_model["answer"])

    return predictions


if __name__ == "__main__":
    data = [
            {"question": "Что умеет машинное обучение",
             "context": "Машинное обучение — это область искусственного интеллекта, "
                         "которая фокусируется на создании моделей, способных обучаться на данных."
                         "Одной из самых распространенных задач является классификация."
                         "В классификации модель обучается на примерах, чтобы уметь определять, "
                         "к какому классу принадлежит новые данные."},
            {"question": "Кто живет на дне океана", "context": "На дне окена живет спанч боб и белка, но это не точно"}
    ]
    print(predict_qa(data))
