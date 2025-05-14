import torch
from transformers import AutoTokenizer, BertForQuestionAnswering, logging

from model_en_ru.translator import *
from congigs_dir.get_configs import Paths

logging.set_verbosity_error()


def get_model_tokenizer():
    """Get tokenizer and model using name model from configs: model_name_en_ru_qa"""
    path_model = Paths().get_model_en_ru()

    tokenizer = AutoTokenizer.from_pretrained(path_model, ignore_mismatched_sizes=True)
    model = BertForQuestionAnswering.from_pretrained(path_model, ignore_mismatched_sizes=True)
    return tokenizer, model


def predict_en_qa(question, text):
    """Get prediction model in ENGLISH"""
    tokenizer, model = get_model_tokenizer()

    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start_index = torch.argmax(start_scores)
    answer_end_index = torch.argmax(end_scores)

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

    if answer is not None:
        return answer
    return 'None'


def predict_from_ru_en(question_ru, text_ru):
    # Translate data to english
    question_en, text_en = translate_ru_en(question_ru), translate_ru_en(text_ru)

    # Get prediction in english
    answer_en = predict_en_qa(question_en, text_en)

    # Translate prediction in russia
    answer_ru = translate_en_ru(answer_en)

    return answer_ru

if __name__ == '__main__':
    question = "Ты кто?"
    context = "Я dragon"
    print(predict_from_ru_en(question, context))
