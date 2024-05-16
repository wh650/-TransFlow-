import os
os.chdir('./data')

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 读取数据
batch_size = 5000
data_chunks = pd.read_csv('selected_proteins100_400stuff.csv', chunksize=batch_size)

# 数据预处理
max_seq_length = 50

tokenizer = Tokenizer(char_level=True)
family_labels = []
sequences = []

for chunk in data_chunks:
    for index, row in chunk.iterrows():
        family_labels.append(row['family_label'])
        seq = row['protein_sequence'][:max_seq_length]
        seq = seq.ljust(max_seq_length, 'X')
        sequences.append(seq)

tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)
sequences = pad_sequences(sequences, maxlen=max_seq_length)

label_to_index = {label: index for index, label in enumerate(np.unique(family_labels))}

labels = [label_to_index[label] for label in family_labels]

X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

X_train_tf = tf.constant(X_train)
X_val_tf = tf.constant(X_val)
X_test_tf = tf.constant(X_test)
y_train_tf = tf.constant(y_train)
y_val_tf = tf.constant(y_val)
y_test_tf = tf.constant(y_test)

# 设置Embedding层为64，LSTM每层256个单元，3层LSTM，epochs为10
embedding_dim = 64
lstm_units = 256
lstm_layers = 3
epochs = 10

# 构建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_seq_length),
])

for i in range(lstm_layers):
    model.add(tf.keras.layers.LSTM(lstm_units, return_sequences=i < lstm_layers - 1))

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(len(label_to_index), activation='softmax'))



# 优化器和学习率调参
optimizers = ['adam', 'rmsprop', 'nadam']
learning_rates = [0.01, 0.001, 0.0001]

for opt in optimizers:
    for lr in learning_rates:
        model_opt = tf.keras.models.clone_model(model)
        model_opt.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(f"\nTraining with optimizer: {opt} and learning rate: {lr}")
        model_opt.fit(X_train_tf, y_train_tf, validation_data=(X_val_tf, y_val_tf), epochs=epochs, batch_size=512)

        # 在测试集上评估模型
        loss, accuracy = model_opt.evaluate(X_test_tf, y_test_tf)
        print(f'Test loss: {loss}')
        print(f'Test accuracy: {accuracy}')
