import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def visualize_pdf_blocks(pdf_path, page_number):
    # Открываем PDF-файл
    doc = fitz.open(pdf_path)
    page = doc[page_number]  # Выбираем страницу по номеру (0-индексация)

    # Получаем изображение страницы
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))

    # Получаем блоки текста
    blocks = page.get_text("blocks")

    # Отображаем страницу
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, aspect='auto')

    # Накладываем прямоугольники на блоки текста
    for block in blocks:
        x0, y0, x1, y1 = block[:4]
        width = x1 - x0
        height = y1 - y0
        rect = Rectangle((x0, pix.height - y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Убираем оси для наглядности
    ax.axis("off")
    plt.show()
    doc.close()
# Пример использования
pdf_path = "test.pdf"  # Замените на путь к вашему PDF
page_number = 0  # Номер страницы, которую хотите визуализировать
visualize_pdf_blocks(pdf_path, page_number)
