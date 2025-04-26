import os
import json
from datasets import Dataset, DatasetDict


def is_path(path):
    if not os.path.exists(path):
        print(f"Wrong path : {path}")
        quit(0)


def read_data_json(path):
    is_path(path)

    with open(path, 'r', encoding='utf8') as data:
        all_data = json.load(data)
    return all_data


def split_data_train_validation(all_data, train_rate=0.8, validation_rate=0.2):
    train_data = all_data[:int(len(all_data) * train_rate)]
    validation_data = all_data[-int(len(all_data) * validation_rate):]

    return train_data, validation_data


def transform_data_to_DatasetDict(train_data, validation_data):
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })

    return dataset_dict


def get_DatasetDict_from_json(path):
    all_data = read_data_json(path)
    train_data, validation_data = split_data_train_validation(all_data)
    dataset_dict = transform_data_to_DatasetDict(train_data, validation_data)
    return dataset_dict


if __name__ == '__main__':
    path = "data/json/IT_data_textbooks.json"
    get_DatasetDict_from_json(path)
