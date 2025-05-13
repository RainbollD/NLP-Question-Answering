import os
import json
import logging


class Paths:
    def __init__(self, config_path=r'.\configs.json'):
        self.config_path = config_path
        self.path_datasets = ''
        self.path_test_data = ''
        self.path_save_test_results = ''
        self.path_model_qa = ''
        self.model_name = ''
        self.path_result_model_qa_fine_tuning = ''
        self.use_local = True
        self.load_config()

    def load_config(self):
        self.is_path(self.config_path)

        with open(self.config_path, 'r', encoding='utf8') as file:
            data_configs = json.load(file)

        self.path_datasets = data_configs.get('path_datasets', '')
        self.path_test_data = data_configs.get('path_test_data', '')
        self.path_save_test_results = data_configs.get('path_save_test_results', '')
        self.path_model_qa = data_configs.get('path_model_qa', '')
        self.model_name = data_configs.get('model_name_qa', '')
        self.path_result_model_qa_fine_tuning = data_configs.get('path_result_model_qa_fine_tuning', '')
        self.use_local = True if data_configs.get('use_local', '') == "True" else False

    @staticmethod
    def is_path(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {path} не найден.")

    def get_datasets_path(self):
        """Path for data"""
        return self.path_datasets

    def get_path_test_data(self):
        """Path to data test"""
        return self.path_test_data

    def get_path_save_test_results(self):
        """Path to save results"""
        return self.path_save_test_results

    def get_model_qa_path(self):
        """Path to model"""
        if self.use_local:
            return self.path_model_qa
        return self.model_name

    def get_model_name(self):
        """Model name"""
        return self.model_name

    def get_path_result_model_qa_fine_tuning(self):
        """Path to fine-turing model"""
        return self.path_result_model_qa_fine_tuning


if __name__ == "__main__":
    try:
        paths_instance = Paths()
        print(paths_instance.get_datasets_path())
        print(paths_instance.get_path_test_data())
        print(paths_instance.get_model_qa_path())
        print(paths_instance.get_model_name())
    except Exception as e:
        logging.error(f"Ошибка: {e}")
