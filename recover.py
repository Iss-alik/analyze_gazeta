from logic import *
#import ollama
import time
import cohere


#token = "OX6F8JL1Cp5rnARJeSZgKRysCehEeCGzJZldVm10"
#token = "AlyMAi6MUQIXXTVuwkEelLdCJds9wgLrwsxDNonM"
token = "zPmJKM0r3f8Y2n9Y2mSj7RAY5GV3ok6utwjs5ioD" #Production key

co = cohere.ClientV2(token)

MODEL_NAME = "mistral"
MODEL_OPTIONS = {
    "temperature": 0.3
    #"top_p": 0.9
}

# Системное сообщение (контекст)
system_message = """ ### Задание и контекст  
Вы работаете с текстом, полученным из отсканированного выпуска газеты. Из-за ошибок OCR текст может содержать:  
1) Опечатки (например, "поеый" → "полный")  
2) Неправильные разрывы слов (например, "Всесо-юзный" → "Всесоюзный")  
3) Ошибки в раскладке клавиатуры (например, "Государство" → "Посударство")  
4) Спутанные буквы из-за схожести символов (например, "О" и "0", "л" и "1", "с" и "с")  

### Ваша задача:  
1) Исправьте ошибки распознавания, восстановив правильное написание слов.  
2) Соедините разорванные слова, если перенос строки или пробелы разделили одно слово.  
3) Исправьте ошибки раскладки (замены русских букв на английские и наоборот).  
4) Сохраните оригинальный стиль и формат текста.  
5) Не изменяйте смысл текста, не добавляйте ничего от себя.  
6) Не сокращайте, не пересказывайте и не упрощайте текст.  
7) Сохраняйте пунктуацию оригинального текста.  

### Формат вывода:  
Выводите **только исправленный текст** без комментариев, пояснений и изменений структуры.  

**Пример входных данных:**  
'1936 года, после в с е н а р о д н о г о обсуждения Чрезвычайный VIII Всесо-юзный с’сзд Советов при-  
нял поеый основ пои закон советского государства — Конституцию С С С Р .. в ОтечестоеппоП соГше осповапа прежде'  

**Пример выходных данных:**  
'1936 года, после всенародного обсуждения Чрезвычайный VIII Всесоюзный съезд Советов принял полный основной закон советского государства — Конституцию СССР, в Отечественной войне основываясь прежде.'  

"""


def recover(raw_text, model=MODEL_NAME, options=MODEL_OPTIONS):
    """Исправляет ошибки OCR в тексте, отправляя его в LLM через Ollama."""
    try:
        start = time.time()

        response = ollama.chat(
            model=model,
            options=options,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": raw_text}
            ]
        )

        restored_text = response.get("message", {}).get("content", "").strip()

        if not restored_text:
            print("[Ошибка] Пустой ответ от модели.")
            return None

        print(f"[Успешно] Время ответа: {time.time() - start:.2f} сек")
        return restored_text

    except Exception as e:
        print(f"[Ошибка] Не удалось обработать текст:\n{raw_text[:300]}...")
        print(f"[Причина] {e}")
        return None
    

def recover2(raw_text):
    try:
        response = co.chat(
        model = "command-a-03-2025",
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": raw_text}
        ],)

        result = response.message.content[0].text

        return result

    except Exception as e:
        print(f"[Ошибка] Не удалось обработать текст:\n{raw_text[:300]}...")
        print(f"[Причина] {e}")
        return raw_text