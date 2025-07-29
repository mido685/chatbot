import joblib
from sentence_transformers import SentenceTransformer
import time 
class IntentClassifier:
    def __init__(self, svm_model_path, label_encoder_path, embedding_model_name='paraphrase-MiniLM-L3-v2'):
        # Load the SVM model and label encoder once during initialization
        self.clf = joblib.load(svm_model_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.embedding_model = SentenceTransformer(embedding_model_name,device='cpu')

    def predict_intent(self, user_input):
        # Encode user input
        embedding = self.embedding_model.encode([user_input])

        # Predict label index
        label_idx = self.clf.predict(embedding)

        # Decode label
        label = self.label_encoder.inverse_transform(label_idx)

        # Get confidence if possible
        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(embedding)[0]
            confidence = max(probs)
        else:
            confidence = 1.0  # fallback

        return label[0], confidence
if __name__ == "__main__":
    # Example usage
    intent_clf = IntentClassifier(
        'models/logistic_classifier.joblib',
        'models/label_encoder2.joblib')
    user_text = "How many chairs are there in branch A?"
    start_time = time.perf_counter()
    intent, confidence = intent_clf.predict_intent(user_text)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Predicted intent: {intent} with confidence {confidence:.2f}")
    print(f"Inference time: {elapsed_time:.6f} seconds")
