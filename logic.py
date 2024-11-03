import nltk
# Splitting Text into Sentences
def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def parts_to_chunks(parts, number_of_chunks):
  numbers_of_parts = len(parts)
  length_of_chunk = int(numbers_of_parts/number_of_chunks) 
  chunks = []
  for x in range(number_of_chunks):
    chunk = " "
    for y in range(length_of_chunk):
      index = y + length_of_chunk * x
      chunk += parts[index]
    chunks.append(chunk)
  return chunks

def chunks_to_dictionary(chunks):
  document_chunked = []

  for chunk in chunks:
    dictinoary_of_chunk = {"data": {"text": chunk}}
    document_chunked.append(dictinoary_of_chunk)
  return document_chunked

def parts_to_dictionary(parts, number_of_chunks = 8):
  chunks = parts_to_chunks(parts, number_of_chunks)
  document_chunked = chunks_to_dictionary(chunks)
  return document_chunked