import os
os.chdir('./data')

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 读取数据
batch_size = 1000
data_chunks = pd.read_csv('selected_proteins5000_400stuff.csv', chunksize=batch_size)

# 数据预处理
max_seq_length_values = list(range(10, 96, 5))  # 截取长度从10到95，步长5
results = []

for max_seq_length in max_seq_length_values:
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

    # 转换为 TensorFlow 张量
    X_train_tf = tf.constant(X_train)
    X_val_tf = tf.constant(X_val)
    X_test_tf = tf.constant(X_test)
    y_train_tf = tf.constant(y_train)
    y_val_tf = tf.constant(y_val)
    y_test_tf = tf.constant(y_test)

    # 设置 Embedding 层为 64
    embedding_dim = 64

    # 构建模型
    input_layer = tf.keras.layers.Input(shape=(max_seq_length,))
    embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_seq_length)(input_layer)
    transformer_block = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=embedding_dim)(embedding, embedding)
    transformer_block = tf.keras.layers.Dense(256, activation='relu')(transformer_block)
    transformer_block = tf.keras.layers.Dropout(0.3)(transformer_block)
    transformer_block = tf.keras.layers.LayerNormalization(epsilon=1e-6)(transformer_block)

    lstm_output = tf.keras.layers.LSTM(256, return_sequences=True)(transformer_block)  # 将 Transformer 的输出作为 LSTM 的输入
    lstm_output = tf.keras.layers.LSTM(256, return_sequences=True)(lstm_output)
    lstm_output = tf.keras.layers.LSTM(256)(lstm_output)

    output = tf.keras.layers.Dense(128, activation='relu')(lstm_output)
    output = tf.keras.layers.Dropout(0.3)(output)
    output = tf.keras.layers.Dense(len(label_to_index), activation='softmax')(output)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)

    # 编译模型
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001), metrics=['accuracy'])

    # 训练模型
    epochs = 10
    batch_size = 256
    history = model.fit(X_train_tf, y_train_tf, validation_data=(X_val_tf, y_val_tf), epochs=epochs, batch_size=batch_size, verbose=0)

    # 在测试集上评估模型
    loss, accuracy = model.evaluate(X_test_tf, y_test_tf, verbose=0)
    results.append((max_seq_length, history.history, loss, accuracy))
    print(f"Max sequence length: {max_seq_length}, Test loss: {loss}, Test accuracy: {accuracy}")

# 输出结果
for result in results:
    max_seq_length, history, loss, accuracy = result
    print(f"Max sequence length: {max_seq_length}, Test loss: {loss}, Test accuracy: {accuracy}")
