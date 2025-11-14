import os
import numpy as np
from gensim.models import KeyedVectors

def test_word2vec_model(kv_path, model_name):
    """
    Test a Word2Vec model by checking vocabulary, vector dimensions, 
    and finding similar words for common test words.
    """
    print(f"\n=== Testing {model_name} Word2Vec Model ===")
    
    # Load the KeyedVectors
    try:
        wv = KeyedVectors.load(kv_path)
        print(f"✓ Successfully loaded {model_name} model from: {kv_path}")
    except FileNotFoundError:
        print(f"✗ Model file not found: {kv_path}")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Test basic model properties
    print(f"📊 Vocabulary size: {len(wv)} words")
    print(f"📏 Vector dimensions: {wv.vector_size}")
    
    # Test words to check similarity
    test_words = ["minister", "prime", "government", "family", "policy"]
    
    print(f"\n🔍 Testing word similarities:")
    for word in test_words:
        if word in wv:
            print(f"\n'{word}' - Top 5 most similar words:")
            try:
                similar = wv.most_similar(word, topn=5)
                for i, (sim_word, score) in enumerate(similar, 1):
                    print(f"  {i}. {sim_word} (similarity: {score:.3f})")
            except Exception as e:
                print(f"  Error finding similar words: {e}")
        else:
            print(f"\n'{word}' - ✗ Not found in vocabulary")
    
   

if __name__ == "__main__":
    # Test both models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Test lemmatized model
    lemmas_kv_path = os.path.join(base_dir, "vectors", "word2vec_lemmas", "word2vec.kv")
    test_word2vec_model(lemmas_kv_path, "Lemmatized")
    
    # Test clean words model
    clean_kv_path = os.path.join(base_dir, "vectors", "word2vec_words", "word2vec.kv")
    test_word2vec_model(clean_kv_path, "Clean Words")