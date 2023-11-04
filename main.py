import requests
import json
from gensim.models import KeyedVectors
from collections import Counter
import jieba
import csv


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

    words_with_lapses = []
    for note in note_info['result']:
        note_fields = note['fields']
        traditional_word = note_fields['Traditional']['value']
        simplified_word = note_fields['Simplified']['value'] if note_fields['Simplified']['value'] else None
        # Extracting cards associated with each note
        cards = note['cards']
        cards_info = invoke('cardsInfo', cards=cards)
        lapses = sum(card['lapses'] for card in cards_info['result'])

        # Add simplified word to the tuple if it exists
        word_info = (traditional_word, lapses)
        if simplified_word:
            word_info += (simplified_word,)

        words_with_lapses.append(word_info)

    return words_with_lapses


def extract_features(words_with_lapses, model_path):
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    features = []
    for word_info in words_with_lapses:
        word = word_info[0]
        incorrect_count = word_info[1]
        simplified_word = word_info[2] if len(word_info) > 2 else None
        if word in model:
            vec = model[word]
            feature = {
                'word': word,
                'vector': vec.tolist(),
                'frequency': Counter(jieba.cut(word)),
                'incorrect_count': incorrect_count
            }
            if simplified_word:
                feature['simplified_word'] = simplified_word
            features.append(feature)
    return features


def save_features_to_csv(features, filename):
    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        header = ['word', 'vector', 'character_frequency', 'incorrect_count']
        # Check if simplified words are in the features to determine if the column should be added
        if any('simplified_word' in feature for feature in features):
            header.insert(1, 'simplified_word')
        writer.writerow(header)
        for feature in features:
            word = feature['word']
            vector = feature['vector']
            frequency = feature['frequency']
            incorrect_count = feature['incorrect_count']
            frequency_str = '; '.join([f'{char}:{count}' for char, count in frequency.items()])
            row = [word, vector, frequency_str, incorrect_count]
            # Insert simplified word into the row if it exists
            if 'simplified_word' in feature:
                row.insert(1, feature['simplified_word'])
            writer.writerow(row)


if __name__ == '__main__':
    deck_name = "Chinese::HanziCraft Review"
    words_to_review = get_words_to_review(deck_name)
    model_path = "./sgns.merge.word"
    features_dataset = extract_features(words_to_review, model_path)
    save_features_to_csv(features_dataset, 'features_dataset.csv')
