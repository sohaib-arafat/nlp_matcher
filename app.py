import os
import nltk
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

nltk.download('wordnet')
nltk.download('punkt')

app = Flask(__name__)

model_name = 'sentence-transformers/stsb-roberta-large'
model = SentenceTransformer(model_name)

SIMILARITY_THRESHOLD = 0.35

def get_synonyms(word):
    try:
        synonyms = set()
        for syn in nltk.corpus.wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms
    except Exception as e:
        print(f"Error in get_synonyms: {str(e)}")
        return set()

def is_synonym_of_any(tokenized_word, matchers):
    try:
        for matcher in matchers:
            if tokenized_word == matcher or tokenized_word in get_synonyms(matcher):
                return True
        return False
    except Exception as e:
        print(f"Error in is_synonym_of_any: {str(e)}")
        return False

def get_embedding(text):
    try:
        return model.encode(text, convert_to_tensor=True).detach().numpy()
    except Exception as e:
        print(f"Error in get_embedding: {str(e)}")
        return None

def cosine_similarity(embedding1, embedding2):
    try:
        return float(util.pytorch_cos_sim(embedding1, embedding2))
    except Exception as e:
        print(f"Error in cosine_similarity: {str(e)}")
        return 0.0

def match_phrases(input_phrase, matchers):
    try:
        results = {}
        input_embedding = get_embedding(input_phrase)

        if input_embedding is None:
            raise ValueError("Failed to obtain input embedding.")

        for matcher_id, matcher_phrase in matchers.items():
            matcher_embedding = get_embedding(matcher_phrase)

            if matcher_embedding is None:
                raise ValueError(f"Failed to obtain embedding for matcher {matcher_id}.")

            similarity = cosine_similarity(input_embedding, matcher_embedding)

            # Tokenize the matcher phrase
            matcher_words = set(token.lower() for token in nltk.word_tokenize(matcher_phrase))

            if any(is_synonym_of_any(token.lower(), matcher_words) for token in nltk.word_tokenize(input_phrase.lower())):
                results[matcher_id] = True
            else:
                results[matcher_id] = similarity > SIMILARITY_THRESHOLD

        return results
    except Exception as e:
        print(f"Error in match_phrases: {str(e)}")
        return {}

@app.route('/process', methods=['POST'])
def process_input():
    try:
        data = request.get_json()
        input_phrase = data.get('input_phrase')
        matchers = data.get('matchers')

        results = match_phrases(input_phrase, matchers)

        response = {
            "success": True,
            "results": results
        }

        return jsonify(response), 200

    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e)
        }
        return jsonify(error_response), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
