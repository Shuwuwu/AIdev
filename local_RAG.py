import os
import requests
import json
import fitz
from tqdm.auto import tqdm
import random

pdf_path = "Short-stories-Sherwood-Anderson-The-Dumb-Man.pdf"
if not os.path.exists(pdf_path):
    print("File doesn't exist, downloading...")
    url = "https://theshortstory.co.uk/devsitegkl/wp-content/uploads/2015/06/Short-stories-Sherwood-Anderson-The-Dumb-Man.pdf"
    filename = pdf_path
    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"The file has been downloaded and saved as {filename}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
else:
    print(f"File {pdf_path} exists.")

def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate pages
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number,
                                "text": text})
    return pages_and_texts

def jaccard_similarity(query, document):
    query = query.lower().split()
    document = document.lower().split()
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union) if union else 0

def return_response(query, corpus):
    similarities = []
    for doc in corpus:
        similarity = jaccard_similarity(query, doc['text'])
        similarities.append(similarity)
    return corpus[similarities.index(max(similarities))] if similarities else None

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)

# user input(for test)
user_input = "wonderful"

# Find the most relevant page based on user input
relevant_page = return_response(user_input, pages_and_texts)
if relevant_page:
    print(f"Most relevant page number: {relevant_page['page_number']}")
    print(f"Text from the most relevant page: {relevant_page['text'][:100]}")
else:
    print(f"No relevant page found.")


api_token = "sk-stgpkrdbpfjbskbezzdcstqvahgjeztdewwxlphcrnzrrjrd"

url = "https://api.siliconflow.cn/v1/embeddings"
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {api_token}"
}

payload = {
    "text": api_token
}

response = requests.post(url, headers=headers, json=payload)

print(response.text)



# Retrival 

# Augmented

# Generation