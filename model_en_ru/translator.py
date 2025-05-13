import translators as ts


def translate_en_ru(text):
    return ts.translate_text(query_text=text, from_language='en', to_language='ru', translator='yandex')


def translate_ru_en(text):
    return ts.translate_text(query_text=text, from_language='ru', to_language='en', translator='yandex')


if __name__ == '__main__':
    en_text = 'Привет мир'
    print(translate_ru_en(en_text))
