#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentAnalyzer:
    def __init__(self, max_length=100, max_words=10000):
        self.max_length = max_length
        self.max_words = max_words
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.label_encoder = LabelEncoder()
        self.model = None

    def prepare_data(self, df):
        df = df.dropna(subset=['text', 'sentiment'])  # Drop rows with missing values
        X = df['text'].astype(str)
        y = self.label_encoder.fit_transform(df['sentiment'])  # Encode labels
        self.tokenizer.fit_on_texts(X)
        X_seq = pad_sequences(self.tokenizer.texts_to_sequences(X), maxlen=self.max_length, padding='post')
        return train_test_split(X_seq, y, test_size=0.2, random_state=42)

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.max_words, 128, input_length=self.max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, df, epochs=1, batch_size=32):
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        self.model = self.create_model()
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                 epochs=epochs, batch_size=batch_size, verbose=1)
        return history

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        seq = pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.max_length, padding='post')
        preds = self.model.predict(seq)
        labels = self.label_encoder.inverse_transform(preds.argmax(axis=1))
        conf = preds.max(axis=1)
        return list(zip(labels, conf))

# Load data and train
df = pd.read_csv(r'C:\Users\felin\Downloads\Sentiment analysis_Social media post.zip')
analyzer = SentimentAnalyzer(max_length=100, max_words=10000)

# Train the model
analyzer.train(df, epochs=1)

# Predictions
texts = ["This game is amazing!", "The service was terrible", "It's okay, nothing special"]
predictions = analyzer.predict(texts)
for text, (sentiment, confidence) in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")


# In[ ]:


This code implements a **sentiment analysis system** using Python, TensorFlow, and Keras. It trains a deep learning model on a dataset of text reviews or comments to classify them into predefined sentiment categories (e.g., positive, neutral, negative). Below is a detailed blockwise explanation:

---

### **1. Importing Required Libraries**

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

- **`pandas`**: Handles data in tabular form (CSV files, data cleaning, and preprocessing).
- **`numpy`**: Provides numerical operations.
- **`tensorflow`**: Used to build, train, and evaluate the deep learning model.
- **`train_test_split`**: Splits the data into training and testing sets.
- **`LabelEncoder`**: Encodes categorical labels into numerical values.
- **`Tokenizer` and `pad_sequences`**: Tokenize text into numerical sequences and pad them to uniform length for input to the neural network.

---

### **2. Class `SentimentAnalyzer`**

This class encapsulates all the functionality for data preparation, model creation, training, and prediction.

#### **Attributes**
```python
def __init__(self, max_length=100, max_words=10000):
    self.max_length = max_length
    self.max_words = max_words
    self.tokenizer = Tokenizer(num_words=self.max_words)
    self.label_encoder = LabelEncoder()
    self.model = None
```

- **`max_length`**: Maximum length of input sequences after padding.
- **`max_words`**: Maximum vocabulary size for tokenization.
- **`tokenizer`**: Converts text into sequences of integers based on word frequency.
- **`label_encoder`**: Encodes categorical labels into numeric form.
- **`model`**: Placeholder for the neural network.

---

#### **3. Data Preparation (`prepare_data`)**

```python
def prepare_data(self, df):
    df = df.dropna(subset=['text', 'sentiment'])  # Drop rows with missing values
    X = df['text'].astype(str)
    y = self.label_encoder.fit_transform(df['sentiment'])  # Encode labels
    self.tokenizer.fit_on_texts(X)
    X_seq = pad_sequences(self.tokenizer.texts_to_sequences(X), maxlen=self.max_length, padding='post')
    return train_test_split(X_seq, y, test_size=0.2, random_state=42)
```

- **Purpose**: Prepares the dataset for training and testing.
- **Steps**:
  1. **Drop Missing Values**: Ensures the dataset has no `NaN` entries in the `text` or `sentiment` columns.
  2. **Text Processing**:
     - Convert the `text` column to strings (if not already).
     - Tokenize the text using the `Tokenizer` to convert words into integers.
     - Pad the sequences to ensure uniform length for all inputs.
  3. **Label Encoding**: Converts sentiment labels (e.g., "positive", "neutral", "negative") into integers.
  4. **Train-Test Split**: Splits data into training (80%) and testing (20%) sets.

---

#### **4. Model Creation (`create_model`)**

```python
def create_model(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(self.max_words, 128, input_length=self.max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

- **Architecture**:
  1. **Embedding Layer**:
     - Maps words to dense vectors of size `128`.
     - Allows the model to learn semantic relationships between words.
  2. **Bidirectional LSTM**:
     - Processes sequences in both forward and backward directions.
     - Captures contextual relationships in text.
  3. **Dense Layer**:
     - Fully connected layer with `64` neurons and `ReLU` activation for non-linear learning.
  4. **Dropout**:
     - Prevents overfitting by randomly dropping 50% of neurons during training.
  5. **Output Layer**:
     - Number of neurons matches the number of sentiment classes.
     - Uses `softmax` activation to output probabilities for each class.

- **Compilation**:
  - **Optimizer**: `Adam` for efficient learning.
  - **Loss Function**: `sparse_categorical_crossentropy` for multi-class classification.
  - **Metric**: Accuracy.

---

#### **5. Training (`train`)**

```python
def train(self, df, epochs=1, batch_size=32):
    X_train, X_test, y_train, y_test = self.prepare_data(df)
    self.model = self.create_model()
    history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                             epochs=epochs, batch_size=batch_size, verbose=1)
    return history
```

- **Purpose**: Trains the model on the dataset.
- **Steps**:
  1. **Data Preparation**: Splits the data into training and testing sets.
  2. **Model Creation**: Builds a new model.
  3. **Training**:
     - Fits the model on the training data for the specified number of epochs and batch size.
     - Evaluates on the testing data after each epoch to monitor performance.

---

#### **6. Prediction (`predict`)**

```python
def predict(self, texts):
    if isinstance(texts, str):
        texts = [texts]
    seq = pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.max_length, padding='post')
    preds = self.model.predict(seq)
    labels = self.label_encoder.inverse_transform(preds.argmax(axis=1))
    conf = preds.max(axis=1)
    return list(zip(labels, conf))
```

- **Purpose**: Predicts sentiments for new text inputs.
- **Steps**:
  1. Converts single or multiple text inputs into tokenized and padded sequences.
  2. Predicts probabilities for each sentiment class.
  3. Retrieves the class with the highest probability and its confidence score.

---

### **7. Execution**

#### **Loading Data and Training**
```python
df = pd.read_csv(r'C:\Users\felin\Downloads\Sentiment analysis_Social media post.zip')
analyzer = SentimentAnalyzer(max_length=100, max_words=10000)
analyzer.train(df, epochs=1)
```

- Loads the dataset (`df`) and trains the model for `1` epoch.

---

#### **Prediction**
```python
texts = ["This game is amazing!", "The service was terrible", "It's okay, nothing special"]
predictions = analyzer.predict(texts)
for text, (sentiment, confidence) in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")
```

- Uses the trained model to predict sentiments for example texts:
  - **Input**: Text like "This game is amazing!"
  - **Output**: Sentiment and confidence for each text.

---

### **Output Explanation**

```plaintext
Text: This game is amazing!
Sentiment: neutral (Confidence: 0.40)

Text: The service was terrible
Sentiment: neutral (Confidence: 0.40)

Text: It's okay, nothing special
Sentiment: neutral (Confidence: 0.40)
```

- **Neutral Output**:
  - The model's performance is poor because:
    1. The training dataset might be too small or noisy.
    2. The model may not have trained long enough (only `1` epoch).
    3. The sentiments in the input examples might be ambiguous.

---

### **Overall Workflow**
1. **Prepare Data**: Clean, tokenize, and split the dataset.
2. **Build Model**: Create and compile an LSTM-based neural network.
3. **Train Model**: Learn from the training data.
4. **Predict Sentiments**: Classify new text inputs and provide confidence scores.

This setup is a basic sentiment analysis pipeline and can be improved with better preprocessing, more training epochs, and hyperparameter tuning.

