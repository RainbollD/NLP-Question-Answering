from use_model_question_answer import predict_qa
import json
import csv
from get_configs import Paths


def read_json():
    test_data_path = Paths().get_path_test_data()
    with open(test_data_path, 'r', encoding='utf8') as file:
        test_data = json.load(file)
    return test_data


def compare_answers(answer_test, answer_model):
    """
    Compare words in each sentence
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
    """Save answer_test answers_model percentages to csv"""
    with open(Paths().get_path_save_test_results(), 'w', newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['question', 'text', 'answer_test', 'answer_model', 'percentage'])

        for answer_test, answer_model, percentage in zip(tests_data, answers_model, percentages):
            writer.writerow([*answer_test.values(), answer_model, str(percentage)])


def test():
    tests_data = read_json()

    answers_model = predict_qa(tests_data)

    match_percentages = []

    for answer_model, test_data in zip(answers_model, tests_data):
        answer_test = test_data["answer"]
        match_percentages.append(compare_answers(answer_test, answer_model))

    save_results_csv(tests_data, answers_model, match_percentages)


if __name__ == "__main__":
    test()
