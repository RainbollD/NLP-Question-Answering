import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, logging

logging.set_verbosity_error()


class Models:
    def __init__(self, model_name = None):
        self.model_name = model_name

    def get_model_tokenizer(self):
        """Get tokenizer and model using model name"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)

        return tokenizer, model

    def predict_tokenizer(self, question, text):
        """Get prediction model in ENGLISH"""
        tokenizer, model = self.get_model_tokenizer()

        inputs = tokenizer(question, text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        answer_start_index = torch.argmax(start_scores)
        answer_end_index = torch.argmax(end_scores) + 1

        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index]
        answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

        return answer if answer else 'None'

    def get_model_pipeline(self):
        if self.model_name is not None:
          return pipeline("question-answering", model=self.model_name)
        return pipeline("question-answering")

    def predict(self, question, text):
        model = self.get_model_pipeline()
        return model(question=question, context=text)['answer'].strip()

if __name__ == '__main__':
    question = "Как тебя зовут?"
    context = "Меня зовут никто"
    try:
        # question_en = translate_ru_en(question)
        # context_en = translate_ru_en(context)
        print(f"Translated question: {question}")
        print(f"Translated context: {context}")

        model = Models('TIGER-Lab/general-verifier')
        prediction = model.predict(question, context)
        # prediction = translate_en_ru(prediction)
        print("Answer:", prediction)
    except Exception as e:
        print(f"Error: {e}")
