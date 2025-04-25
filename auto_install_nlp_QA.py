from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from get_configs import Paths

model_name = Paths().get_model_name()
path_model_save = Paths().get_model_qa_path()

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(path_model_save)
tokenizer.save_pretrained(path_model_save)
