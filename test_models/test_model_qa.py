import json
import csv
import os

from model_rus.use_model_qa_ru import predict_ru_qa
from get_configs import Paths


def read_json():
    test_data_path = Paths().get_path_test_data()
    with open(test_data_path, 'r', encoding='utf8') as file:
        test_data = json.load(file)
    return test_data


def compare_answers(answer_test, answer_model):
    """
    Counts the same number of words and
    divides by the length of the unique characters of the answers in answer_test
    :param answer_test:
    :param answer_model:
    :return:
    """
    answer_test = answer_test.lower().strip()
    answer_model = answer_model.lower().strip()

    words1 = set(answer_test.split())
    words2 = set(answer_model.split())

    matching_words = words1.intersection(words2)

    total_unique_words = len(words1.union(words2))

    if total_unique_words == 0:
        return 0.0

    match_percentage = (len(matching_words) / total_unique_words) * 100

    return match_percentage


def save_results_csv(tests_data, answers_model, percentages):
    """Save question, text, answer, model answer, percentage to csv"""
    with open(os.path.join(Paths().get_path_save_test_results(), ), 'w', newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['question', 'text', 'answer_test', 'answer_model', 'percentage'])

        for answer_test, answer_model, percentage in zip(tests_data, answers_model, percentages):
            writer.writerow([*answer_test.values(), answer_model, str(percentage)])


def test():
    tests_data = read_json()

    answers_model = []

    for test_data in tests_data:

        question = test_data['question']
        context = test_data['context']

        prediction = predict_ru_qa(question, context)
        answers_model.append(prediction)

    match_percentages = []

    for answer_model, test_data in zip(answers_model, tests_data):
        answer_test = test_data["answer"]
        match_percentages.append(compare_answers(answer_test, answer_model))

    save_results_csv(tests_data, answers_model, match_percentages)


if __name__ == "__main__":
    test()
