import random
import requests
import json
import os

key = 'sk-1lw82c7e44b2245f810fb1ece98153343106c8972e09Jcsc'

def embedding_retrieve(term):
    # Set up the API endpoint URL and request headers
    # url = "https://api.openai.com/v1/embeddings"
    url = "https://api.gptsapi.net/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    # Set up the request payload with the text string to embed and the model to use
    payload = {
        "input": term,
        # "model": "text-embedding-3-large"
        "model": "text-embedding-3-small"
    }

    # Send the request and retrieve the response
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Extract the text embeddings from the response JSON
    embedding = response.json()["data"][0]['embedding']

    return embedding

def get_random_embedding(term):
    return [random.uniform(-1, 1) for _ in range(1536)]

if __name__ == '__main__':
    t = embedding_retrieve(term="A patient has a cold no no no.")
    tt = get_random_embedding()
    print(1)