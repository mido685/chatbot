# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.svm import SVC
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report,confusion_matrix
# import matplotlib.pyplot as plt
# import joblib

# # Load your CSV
# df = pd.read_csv("config/combined_intents.csv")
# # to check if the data is balanced or not 
# label_counts=df['label'].value_counts()
# # to check the ratio of imbalanced data
# imbalance_ratio=label_counts.max()/label_counts.min()
# # Load sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')  # Good small model

# # Convert text to embeddings
# X = model.encode(df['text'].tolist())

# # Encode labels
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df['label'])
# # label_counts.plot(kind="bar", title="Class Distribution")
# # plt.ylabel("Number of Samples")
# # plt.xlabel("Class Label")
# # plt.xticks(rotation=45)
# # plt.tight_layout()
# # plt.show()


# # Split into train/test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train SVM classifier
# clf = SVC(probability=True)
# clf.fit(X_train, y_train)

# # Make predictions
# y_pred = clf.predict(X_test)

# # Evaluate the model
# classification_report=classification_report(y_test, y_pred, target_names=label_encoder.classes_)
# confusion_matrix=confusion_matrix(y_test,y_pred)

# # Save SVM classifier
# joblib.dump(clf,'svm_classifier.joblib')

# # Save label encoder
# joblib.dump(label_encoder, 'label_encoder.joblib')
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load data
df = pd.read_csv(r"C:\Users\AL_hamd\OneDrive\Desktop\Chatbot\Config\combined_intents.csv")  # your file

# 2. Load fast SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # very fast + small

# 3. Embed text
X = model.encode(df['text'].tolist(), show_progress_bar=False)

# 4. Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train a fast classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. Save models
joblib.dump(clf, 'models/logistic_classifier.joblib')
joblib.dump(label_encoder, 'models/label_encoder2.joblib')
