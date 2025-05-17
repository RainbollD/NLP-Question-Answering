import os
import json
import logging


class Paths:
    def __init__(self, config_path=r'C:\Users\Lev\NLP-Question-Answering\configs_dir\configs.json'):
        self.config_path = config_path
        self.path_test_data = ''
        self.path_save_test_results = ''
        self.model_names = ''
        self.load_config()

    def load_config(self):
        self.is_path(self.config_path)

        with open(self.config_path, 'r', encoding='utf8') as file:
            data_configs = json.load(file)

        self.path_test_data = data_configs.get('path_test_data', '')
        self.path_save_test_results = data_configs.get('path_save_test_results', '')
        self.model_names = data_configs.get('model_names', '')

    @staticmethod
    def is_path(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Файл {path} не найден.")

    def get_path_test_data(self):
        """Path to data test"""
        return self.path_test_data

    def get_path_save_test_results(self):
        """Path to save results"""
        return self.path_save_test_results

    def get_model_names(self):
        """Path to model"""
        return self.model_names


if __name__ == "__main__":
    try:
        paths_instance = Paths()
        print(paths_instance.get_model_names())
        print(paths_instance.get_path_test_data())
        print(paths_instance.get_path_save_test_results())
        print()
    except Exception as e:
        logging.error(f"Ошибка: {e}")
