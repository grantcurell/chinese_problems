import requests
import json
from gensim.models import Word2Vec
from collections import Counter
import jieba
import csv
from gensim.models import KeyedVectors

def invoke(action, **params):
    request_json = json.dumps({
        'action': action,
        'params': params,
        'version': 6
    })
    response = requests.post('http://localhost:8765', data=request_json)
    return json.loads(response.content)

def get_words_to_review(deck_name):
    query = f'"note:Chinese Words Hanzicraft" (is:review OR is:learn)'
    result = invoke('findNotes', query=query)
    note_ids = result.get('result', [])
    note_info = invoke('notesInfo', notes=note_ids)
    return [note['fields']['Traditional']['value'] for note in note_info['result']] if 'result' in note_info else []

def extract_features(words, model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)  # Load your pre-trained Word2Vec model in the correct format
    features = []
    for word in words:
        if word in model:  # Check if the word is in the model vocabulary
            vec = model[word]  # Get the word vector
            features.append(
                {'word': word, 'vector': vec.tolist(), 'frequency': Counter(jieba.cut(word))}
            )  # Add the word, its vector, and frequency data to the features list
    return features

def save_features_to_csv(features, filename):
    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['word', 'vector', 'character_frequency'])
        for feature in features:
            word = feature['word']
            vector = feature['vector']
            frequency = feature['frequency']
            # Flatten frequency counter to a string representation
            frequency_str = '; '.join([f'{char}:{count}' for char, count in frequency.items()])
            writer.writerow([word, vector, frequency_str])  # Write the word, its vector, and frequency data as a row in the CSV

if __name__ == '__main__':
    deck_name = "Chinese::HanziCraft Review"
    words_to_review = get_words_to_review(deck_name)
    model_path = "./sgns.merge.word"
    features_dataset = extract_features(words_to_review, model_path)
    save_features_to_csv(features_dataset, 'features_dataset.csv')  # Save the features to a CSV file named 'features_dataset.csv'