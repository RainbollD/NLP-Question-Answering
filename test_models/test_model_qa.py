import os
import re
import json
import csv

from translator import *
from bert_score import score
from tqdm import tqdm

from use_model_question_answer import Models
from configs_dir.get_configs import Paths


def read_json():
    test_data_path = Paths().get_path_test_data()
    with open(test_data_path, 'r', encoding='utf8') as file:
        test_data = json.load(file)
    return test_data


def compare_answers(real_answer, model_answer):
    P, R, F1 = score([model_answer], [real_answer], lang="ru", model_type="bert-base-multilingual-cased")
    return {"P": P.item(), "R": R.item(), "F1": F1.item()}


def save_results_csv(tests_data, model_answer, match_percentages, addition_name):
    """Save question, text, answer, model answer, percentage to csv"""

    model_name_for_csv = re.split(r'/|\\', addition_name)[0]
    with open(os.path.join(Paths().get_path_save_test_results(), f"{model_name_for_csv}.csv"), 'w', newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['question', 'context', 'answer', 'model_answer', 'match_percentages'])

        for answer_test, answer_model, match_percentage in zip(tests_data, model_answer, match_percentages):
            writer.writerow([*answer_test.values(), answer_model, str(match_percentage)])


def test_model_predictions(model_name, tests_data):
    predictions = []
    model = Models(model_name)
    for test_data in tests_data:
        question_en = translate_ru_en(test_data['question'])
        context_en = translate_ru_en(test_data['context'])

        result_en = model.predict(question_en, context_en)
        result = translate_en_ru(result_en)

        predictions.append(result)
    return predictions


def test_model_match_percentages(model_answers, real_answers):
    match_percentages = []
    for model_answer, real_answer in zip(model_answers, real_answers):
        match_percentages.append(compare_answers(real_answer['answer'], model_answer))
    return match_percentages


def test_control():
    tests_data = read_json()

    #model_names = Paths().get_model_names()
    model_names = ['']
    for model_name in tqdm(model_names):
        predictions = test_model_predictions(model_name, tests_data)
        match_percentages = test_model_match_percentages(predictions, tests_data)
        save_results_csv(tests_data, predictions, match_percentages, model_name.split('\\|/')[0])


if __name__ == "__main__":
    test_control()
