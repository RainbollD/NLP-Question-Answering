import torch
from transformers import AutoTokenizer, BertForQuestionAnswering, logging

from translator import *

logging.set_verbosity_error()


def get_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
                                              ignore_mismatched_sizes=True)
    model = BertForQuestionAnswering.from_pretrained(
        "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad", ignore_mismatched_sizes=True)
    return tokenizer, model


def predict_en_qa(question, text):
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

    return answer


def predict_from_ru_en(question_ru, text_ru):
    question_en, text_en = translate_ru_en(question_ru), translate_ru_en(text_ru)

    answer_en = predict_en_qa(question_en, text_en)

    answer_ru = translate_en_ru(answer_en)

    return answer_ru
