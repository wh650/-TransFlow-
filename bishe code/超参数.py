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
data_chunks = pd.read_csv('selected_proteins300_400stuff.csv', chunksize=batch_size)

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

# 设置参数组合
embedding_dims = [64, 128, 256]  # Embedding层的维度
lstm_units = [64, 128, 256]  # LSTM层每层的单元数
lstm_layers = [1, 2, 3]  # LSTM层的深度
epochs = 10

# 创建结果保存的DataFrame
results_df = pd.DataFrame(columns=['Embedding_dims', 'LSTM_units', 'LSTM_layers', 'Test_loss', 'Test_accuracy'])

for emb_dim in embedding_dims:
    for unit in lstm_units:
        for layer in lstm_layers:
            print(f'Running for: Embedding_dims={emb_dim}, LSTM_units={unit}, LSTM_layers={layer}')
            
            # 构建模型
            model = tf.keras.models.Sequential([
                tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=emb_dim, input_length=max_seq_length),
            ])
            for i in range(layer):
                model.add(tf.keras.layers.LSTM(unit, return_sequences=i < layer - 1))

            model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.Dense(len(label_to_index), activation='softmax'))

            model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

            model.fit(X_train_tf, y_train_tf, validation_data=(X_val_tf, y_val_tf), epochs=epochs)

            # 在测试集上评估模型
            loss, accuracy = model.evaluate(X_test_tf, y_test_tf)
            print(f'Test loss: {loss}')
            print(f'Test accuracy: {accuracy}')

            results_df = pd.concat([results_df, pd.DataFrame({'Embedding_dims': [emb_dim], 'LSTM_units': [unit], 'LSTM_layers': [layer], 'Test_loss': [loss], 'Test_accuracy': [accuracy]})], ignore_index=True)

# 保存结果到csv文件
results_df.to_csv('model_results.csv', index=False)
