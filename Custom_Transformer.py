!pip uninstall -y tensorflow keras-nlp
!pip install tensorflow==2.17 keras-nlp==0.5.0


import tensorflow as tf
from tensorflow.keras.layers import *
import keras_nlp
import re
import os
import keras_nlp
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model
from google.colab import files
import pandas as pd
from tensorflow.keras.layers import Embedding, Dropout, LayerNormalization, Dense, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np

mpolicy = keras.mixed_precision.Policy("mixed_float16")
keras.mixed_precision.set_global_policy(mpolicy)


data = pd.read_csv("train.csv")

# Data preprocessing
data = pd.read_csv("train.csv")
data = data.drop('SampleID', axis=1)

def preprocessing(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.strip()
    return text

data = data.dropna(subset=['Discussion'])
data['Discussion'] = data['Discussion'].astype(str)
data['Discussion'] = data['Discussion'].str.lower()
data['Discussion'] = data['Discussion'].apply(preprocessing)

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['Discussion'])
sequences = tokenizer.texts_to_sequences(data['Discussion'])

max_len = 64  # Reduced length
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', maxlen=max_len)

# Create category mapping
category_map = {"Politics": 0, "Sports": 1, "Media": 2, "Market & Economy": 3, "STEM": 4}
labels = data['Category'].map(category_map).values

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels))
dataset = dataset.shuffle(buffer_size=1024)

# Split dataset
train_size = int(0.8 * len(dataset))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

def build_bert_model(vocab_size, max_len, num_heads, num_layers, d_model, dff, rate=0.1):
    inputs = Input(shape=(max_len,))
    x = Embedding(vocab_size, d_model)(inputs)
    x += positional_encoding(max_len, d_model)

    for _ in range(num_layers):
        x = encoder_layer(d_model, num_heads, dff, rate)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='gelu')(x)  # Reduced dense layer size
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(category_map), activation='softmax')(x)

    return Model(inputs, outputs)


def positional_encoding(max_len, d_model):
    pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rads = pos / tf.pow(10000, (2 * (i // 2)) / d_model)
    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    return tf.expand_dims(pos_encoding, axis=0)

def encoder_layer(d_model, num_heads, dff, rate):
    inputs = Input(shape=(None, d_model))
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(inputs, inputs)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = Dense(dff, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(out1)  # Added L2 Regularization
    ffn_output = Dense(d_model)(ffn_output)  # Correct placement of Dense layer
    ffn_output = Dropout(rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return Model(inputs=inputs, outputs=out2)


# Custom callback for printing accuracy
class PrintAccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}")
        print(f"Training Accuracy: {logs['accuracy']:.4f}")
        print(f"Validation Accuracy: {logs['val_accuracy']:.4f}")

# Calculate class weights to handle slight imbalance
def calculate_class_weights(labels):
    total_samples = len(labels)
    class_counts = np.bincount(labels)
    class_weights = {i: total_samples / (len(class_counts) * count)
                    for i, count in enumerate(class_counts)}
    return class_weights

# Training setup
def create_model_and_train():
    # Model parameters
    vocab_size = len(tokenizer.word_index) + 1
    num_classes = len(category_map)
    num_heads = 4  # Reduced number of heads
    num_layers = 4  # Reduced number of layers
    d_model = 128  # Reduced model size
    dff = 512  # Reduced FFN size
    rate = 0.1

    # Learning rate schedule
    initial_learning_rate = 1e-4  # Lower initial learning rate
    decay_steps = len(train_dataset) // 64 * 2  # Decay every 2 epochs
    decay_rate = 0.9
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )  # Edited part to use learning rate schedule without conflict

    # Create and compile model
    bert_model = build_bert_model(vocab_size, max_len, num_heads, num_layers, d_model, dff, rate)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    bert_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Calculate class weights
    class_weights = calculate_class_weights(labels)
    print("\nClass weights:", class_weights)

    # Prepare datasets with batching and prefetch
    batched_train_dataset = train_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
    batched_val_dataset = val_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=4,  # Increased patience
            restore_best_weights=True,
            min_delta=0.001  # Minimum improvement required
        ),
        ModelCheckpoint(
            'best_model.weights.h5',  # Updated file path
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True
        ),
        PrintAccuracyCallback()
    ]


    # Train the model
    print("\nStarting training...")
    history = bert_model.fit(
        batched_train_dataset,
        epochs=15,
        validation_data=batched_val_dataset,
        callbacks=callbacks,
        class_weight=class_weights
    )

    return bert_model, history

# Train the model
classifier, history = create_model_and_train()

# Final evaluation
print("\nFinal Evaluation:")
test_dataset = val_dataset.batch(64)  # Using validation set as test set for this example
test_loss, test_accuracy = classifier.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print training history summary
print("\nTraining History Summary:")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")