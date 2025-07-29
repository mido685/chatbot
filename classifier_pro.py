import joblib
import time
import numpy as np
import onnxruntime
from transformers import AutoTokenizer

class IntentClassifierONNX:
    def __init__(self, svm_model_path, label_encoder_path, onnx_model_path, tokenizer_name):
        self.clf = joblib.load(svm_model_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    def encode(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="np", padding=True, truncation=True)
        valid_inputs = ['input_ids', 'attention_mask']
        ort_inputs = {k: v for k, v in inputs.items() if k in valid_inputs}

        ort_outputs = self.session.run(None, ort_inputs)
        token_embeddings = ort_outputs[0]  # shape: (batch_size, seq_len, hidden_size)
        attention_mask = ort_inputs['attention_mask']

        # Convert attention mask to shape (batch_size, seq_len, 1)
        mask = attention_mask[..., np.newaxis].astype(np.float32)

        # Apply mean pooling
        summed = np.sum(token_embeddings * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        sentence_embeddings = summed / counts

        return sentence_embeddings  # shape: (batch_size, hidden_size)


    def predict_intent(self, user_input):
        embedding = self.encode([user_input])
        return self._predict_from_embeddings(embedding)[0]

    def predict_batch(self, user_inputs):
        embeddings = self.encode(user_inputs)
        return self._predict_from_embeddings(embeddings)

    def _predict_from_embeddings(self, embeddings):
        label_idxs = self.clf.predict(embeddings)
        labels = self.label_encoder.inverse_transform(label_idxs)

        confidences = []
        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(embeddings)
            confidences = probs.max(axis=1)
        else:
            confidences = [1.0] * len(labels)

        return list(zip(labels, confidences))
if __name__ == "__main__":
    intent_clf = IntentClassifierONNX(
        'models/logistic_classifier.joblib',
        'models/label_encoder2.joblib',
        r'C:\Users\AL_hamd\OneDrive\Desktop\Chatbot\onnx_model\model.onnx',
        'sentence-transformers/all-MiniLM-L6-v2'
    )

    user_texts = [
            "how many tables in branch Golf",
            "how many tables in branch Arkan",
            "hello there",
            "good morning",
            "good evening",
            "evening!",
            "greetings",
            "goodbye",     # should be farewell
            "see you later", 
            "good morning",
            "good evening"
    ]

    start_time = time.perf_counter()
    results = intent_clf.predict_batch(user_texts)
    end_time = time.perf_counter()

    for text, (intent, confidence) in zip(user_texts, results):
        print(f"Input: {text}")
        print(f"Predicted intent: {intent} with confidence {confidence:.2f}")
        print("")

    print(f"Total batch inference time: {end_time - start_time:.6f} seconds")
