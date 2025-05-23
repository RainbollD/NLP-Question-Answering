import os
import re
import json
import csv

import pandas as pd

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


def read_txt(path):
    with open(path, 'r', encoding='utf8') as file:
        file_data = file.read().strip().split('\n')
        return file_data


def compare_answers(real_answer, model_answer):
    P, R, F1 = score([model_answer], [real_answer], lang="ru", model_type="bert-base-multilingual-cased")
    return {"P": P.item(), "R": R.item(), "F1": F1.item()}


def save_results_different_csv(tests_data, model_answer, match_percentages, addition_name):
    """Save question, text, answer, model answer, percentage to csv"""

    model_name_for_csv = re.split(r'/|\\', addition_name)[0]
    with open(os.path.join(Paths().get_path_save_test_results(), f"{model_name_for_csv}.csv"), 'w', newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['question', 'context', 'answer', 'model_answer', 'match_percentages'])

        for answer_test, answer_model, match_percentage in zip(tests_data, model_answer, match_percentages):
            writer.writerow([*answer_test.values(), answer_model, str(match_percentage)])


def save_results_one_csv(model_answer, addition_name):
    """Save question, text, answer, model answer, percentage to csv"""
    if addition_name is None:
        addition_name = 'question-answering'

    df = pd.read_csv("/content/result_all_models.csv")
    df[addition_name] = model_answer
    df.to_csv("/content/result_all_models.csv", index=False)


def test_model_predictions_json(model_name, tests_data):
    predictions = []
    model = Models(model_name)
    for test_data in tests_data:
        question_en = translate_ru_en(test_data['question'])
        context_en = translate_ru_en(test_data['context'])

        result_en = model.predict(question_en, context_en)
        result = translate_en_ru(result_en)

        predictions.append(result)
    return predictions


def test_model_predictions_txt(model_name, context, questions):
    predictions = []
    model = Models(model_name)
    context_en = translate_ru_en(' '.join(context))
    for question in questions:
        question_en = translate_ru_en(question)

        result_en = model.predict(question_en, context_en)
        result = translate_en_ru(result_en)

        predictions.append(result)

    return predictions


def test_model_match_percentages(model_answers, real_answers):
    match_percentages = []
    for model_answer, real_answer in zip(model_answers, real_answers):
        match_percentages.append(compare_answers(real_answer['answer'], model_answer))
    return match_percentages


def is_csv_for_all_results(questions):
    path = r"C:\Users\Lev\NLP-Question-Answering\test_models\vlad_data\results\result_all_models.csv"

    def create_csv_file_if_none():
        """Initializes the CSV file with a header if it doesn't exist."""
        if os.path.exists(path): return

        data = ['Question']
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(data)

    def check_questions(questions):
        """Checks if the questions already exist in the file and writes new ones."""
        questions_in_file = set()
        questions = set(questions)

        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                questions_in_file.add(row[0])

        new_questions = questions - questions_in_file
        if new_questions:
            with open(path, 'a', newline='') as f:
                writer = csv.writer(f)
                for question in new_questions:
                    writer.writerow([question])

    create_csv_file_if_none()
    check_questions(questions)


def test_different_data_control():
    tests_data = read_json()

    model_names = Paths().get_model_names()
    for model_name in tqdm(model_names):
        predictions = test_model_predictions_json(model_name, tests_data)
        match_percentages = test_model_match_percentages(predictions, tests_data)
        save_results_different_csv(tests_data, predictions, match_percentages, model_name.split('\\|/')[0])


def test_vlad_data_control():
    paths = Paths()

    context = read_txt(paths.get_path_vlad_context())
    questions_data = read_txt(paths.get_path_vlad_question())

    model_names = Paths().get_model_names()
    for model_name in tqdm(model_names):
        if model_name == "None": model_name = None

        predictions = test_model_predictions_txt(model_name, context, questions_data)
        save_results_different_csv(context, predictions, [], model_name.replace('\\|/', '_'))


if __name__ == "__main__":
    test_vlad_data_control()
