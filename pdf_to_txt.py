import pytesseract
from PIL import Image
import cohere
from pdf2image import convert_from_path
import os
 
co = cohere.Client('AH0agiBtbhKCBApQrt9OscLj22VjX6JV0O9KBwrN')

Image.MAX_IMAGE_PIXELS = 2000000000


for file in os.listdir(".\pdf_files"):
  file_name = file[0:-4]
  pages = convert_from_path(f'pdf_files/{file}', 500)
  for page in pages:
    page.save('out.png', 'PNG')
    # Open the image file
    image = Image.open('out.png')

    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
    # Perform OCR using PyTesseract
    raw_text = pytesseract.image_to_string(image, lang = 'rus')

    file = open(f'text_files/{file_name}.txt', 'a')
    file.write(raw_text)#.encode('utf8'))
    file.close()


