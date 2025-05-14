from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from congigs_dir.get_configs import Paths

# Prepare name and path to save
model_name = Paths().get_model_name()
path_model_save = Paths().get_model_rus_qa_path()

# Create model and tokenizer for this model
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save model and tokenizer
model.save_pretrained(path_model_save)
tokenizer.save_pretrained(path_model_save)
