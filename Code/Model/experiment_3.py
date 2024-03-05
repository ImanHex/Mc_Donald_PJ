import pickle
from pathlib import Path
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

stored_folder = Path(os.path.abspath('')).parent.parent / "data" / "processed" / "cleaned_df.pkl"
input_file = open(stored_folder, "rb")
cleaned_data = pickle.load(input_file)

if __name__ == "__main__":
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_data['processed_review'])
    y = cleaned_data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    preds = gbc.predict(X_test)
    print(f'Accuracy:  {accuracy_score(y_test, preds)}')

    cm = confusion_matrix(y_test, preds)
    print('Confusion Matrix:')
    print(cm)

    labels = ['Positive', 'Negative', 'Neutral']
    cm = confusion_matrix(y_test, preds, labels=labels)

    # Plotting the confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    plt.title('McDonald`s Sentiment Confusion Matrix(Gradient Boosting)')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    # Accuracy 0.67