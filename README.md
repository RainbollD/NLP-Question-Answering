# NLP: Question Answering

## Описание

NLP QA — модель для ответов на вопросы, учитывая контекст. На вход модели подается текст и вопрос, на основе которыз модель выдает строку с ответом.

## Установка

1. Склонируйте репозиторий:
   ```bash
   https://github.com/RainbollD/NLP-Question-Answering.git
   ```

2. Установите необходимые зависимости:
   ```bash
   pip install -r requirements.txt
   ```
## Использование

Локальный запуск:
- Запустите `auto_install_nlp_QA.py` (размер модели ~ 2 гб)
- В файле `configs.json` установить параметр `use_local = True`

Запуск напрямую:
- В файле `configs.json` установить параметр `use_local = False`

Запустите файл `use_model_question_answer.py`
- Передавать данные в функцию `predict_qa` в формате:  
 [{"question" : "...", ""context" : "..."}, ...]

