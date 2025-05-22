import os
import json
import logging


class Paths:
    def __init__(self, config_path=r'C:\Users\Lev\NLP-Question-Answering\configs_dir\configs.json'):
        self.config_path = config_path
        self.path_test_data = ''
        self.path_save_test_results = ''
        self.model_names = ''
        self.path_vlad_question = ''
        self.path_vlad_context = ''
        self.path_save_vlad_results = ''
        self.load_config()

    def load_config(self):
        self.is_path(self.config_path)

        with open(self.config_path, 'r', encoding='utf8') as file:
            data_configs = json.load(file)

        self.path_test_data = data_configs.get('path_test_data', '')
        self.path_save_test_results = data_configs.get('path_save_test_results', '')
        self.model_names = data_configs.get('model_names', '')
        self.path_vlad_question = data_configs.get('path_vlad_question', '')
        self.path_vlad_context = data_configs.get('path_vlad_context', '')
        self.path_save_vlad_results = data_configs.get('path_save_vlad_results', '')

    def is_path(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {path} не найден.")

    def get_path_test_data(self):
        """Path to data test"""
        self.is_path(self.path_test_data)

        return self.path_test_data

    def get_path_save_test_results(self):
        """Path to save different data results"""
        self.is_path(self.path_save_test_results)

        return self.path_save_test_results

    def get_model_names(self):
        """Path to models"""
        return self.model_names

    def get_path_vlad_question(self):
        """Path to vlad questions test"""
        self.is_path(self.path_vlad_question)

        return self.path_vlad_question

    def get_path_save_vlad_results(self):
        """Path to save vlad results"""
        self.is_path(self.path_save_vlad_results)

        return self.path_save_vlad_results

    def get_path_vlad_context(self):
        """Path to vlad's text"""
        self.is_path(self.path_vlad_context)

        return self.path_vlad_context


if __name__ == "__main__":
    try:
        paths_instance = Paths()
        print(paths_instance.get_model_names())
        print(paths_instance.get_path_test_data())
        print(paths_instance.get_path_save_test_results())
        print(paths_instance.get_path_vlad_context())
        print(paths_instance.get_path_vlad_question())
        print(paths_instance.get_path_save_vlad_results())
        print()
    except Exception as e:
        logging.error(f"Ошибка: {e}")
