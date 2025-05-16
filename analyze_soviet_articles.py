
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import pipeline
from datetime import datetime


# === ПУТЬ К ДАННЫМ ===
TEXT_DIR = "./sample"
OUTPUT_DIR = "./"
PLOT_DIR = "./"


# Функция для чтения всех текстовых файлов из папки
def read_documents_from_folder(folder_path):
    data = []
    counter = 0
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                date_str = file_name.split(".")[0]
                date = datetime.strptime(date_str, "%d-%m-%Y")
                data.append({"filename": file_name, "date": date, "text": text})    
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                counter += 1

    print("Number of unreaded documents: ", counter)
    return pd.DataFrame(data)



# === СЕНТИМЕНТАНАЛИЗ ===
def analyze_sentiment(texts):
    sentiment_pipeline = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment", tokenizer="blanchefort/rubert-base-cased-sentiment")
    sentiments = []
    for text in tqdm(texts):
        try:
            result = sentiment_pipeline(text[:512])[0]  
            sentiments.append(result)
        except:
            print(f"Sentiment analysis failed on index {i}: {e}")
            sentiments.append({"label": "NEUTRAL", "score": 0})
    return sentiments

# === ВИЗУАЛИЗАЦИЯ СЕНТИМЕНТА ===
def plot_sentiment_over_time(df):
    df["month"] = df["date"].dt.to_period("M")
    monthly_sentiment = df.groupby("month")["sentiment_label"].value_counts(normalize=True).unstack().fillna(0)
    monthly_sentiment.plot(kind="line", figsize=(12, 6), marker="o")
    plt.title("Доля сентиментов по месяцам")
    plt.ylabel("Доля")
    plt.xlabel("Месяц")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sentiment_over_time.png"))

# === ОСНОВНОЙ СКРИПТ ===
def main():
    df = read_documents_from_folder(TEXT_DIR)

    # Сентимент-анализ
    sentiments = analyze_sentiment(df["text"])
    df["sentiment_label"] = [s["label"] for s in sentiments]
    df["sentiment_score"] = [s["score"] for s in sentiments]

    df.to_csv(os.path.join(OUTPUT_DIR, "analysis_results.csv"), index=False, encoding="utf-8-sig")
    plot_sentiment_over_time(df)

if __name__ == "__main__":
    main()
