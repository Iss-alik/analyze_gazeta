import pytesseract
from PIL import Image
import cohere
from pdf2image import convert_from_path

co = cohere.Client('AH0agiBtbhKCBApQrt9OscLj22VjX6JV0O9KBwrN')

Image.MAX_IMAGE_PIXELS = 2000000000
pages = convert_from_path('src/gazeta1.pdf', 500)
counter = 0

for page in pages:
    page.save('out.png', 'PNG')
    # Open the image file
    image = Image.open('out.png')

    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
    # Perform OCR using PyTesseract
    raw_text = pytesseract.image_to_string(image, lang = 'rus')


    message = raw_text

    response = co.chat(
    model="command-r-plus",
    chat_history=[
    {"role": "USER", "text": "Я анализирую пдф файлы с помощью компьютерного зрения и достаю от туда текст. Однако точно желает лучшего. Не мог бы ты пожалуйста востановить следующий текст"},
    {"role": "CHATBOT", "text": "Да конечно, вышлите пожалуйста сам текст"}
  ],  
    message = message,
  )
    text = response.text

    file = open(f'{counter}.txt', 'wb')
    file.write(text.encode('utf8'))
    file.close()
    counter += 1

