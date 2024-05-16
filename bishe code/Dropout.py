import os
os.chdir('./data')

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# 读取数据
batch_size = 5000
data_chunks = pd.read_csv('selected_proteins5000_400stuff.csv', chunksize=batch_size)

# 数据预处理
max_seq_length = 48

tokenizer = Tokenizer(char_level=True)
family_labels = []
sequences = []

for chunk in data_chunks:
    for index, row in chunk.iterrows():
        family_labels.append(row['family_label'])
        seq = row['protein_sequence'][:max_seq_length]
        seq = seq.ljust(max_seq_length, 'X')  # 使用字符'X'补齐至30个字符
        sequences.append(seq)

tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)
sequences = pad_sequences(sequences, maxlen=max_seq_length)

# 将家族标签转换为数字编码
label_to_index = {label: index for index, label in enumerate(np.unique(family_labels))}
index_to_label = {index: label for label, index in label_to_index.items()}
labels = [label_to_index[label] for label in family_labels]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 将数据转换为tf.Tensor类型
X_train_tf = tf.constant(X_train)
X_val_tf = tf.constant(X_val)
X_test_tf = tf.constant(X_test)
y_train_tf = tf.constant(y_train)
y_val_tf = tf.constant(y_val)
y_test_tf = tf.constant(y_test)

# 定义不同的Dropout参数
dropout_values = [0.6]

# 存储训练过程中的Loss和Accuracy
with open('training_logs.txt', 'w') as f:
    for dropout_value in dropout_values:
        # 构建模型
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_seq_length),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(dropout_value),  # 调整Dropout参数
            tf.keras.layers.Dense(len(label_to_index), activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

        # 训练模型
        history = model.fit(X_train_tf, y_train_tf, validation_data=(X_val_tf, y_val_tf), epochs=10, batch_size=128)

        # 保存训练过程中的Loss和Accuracy
        f.write(f"Dropout: {dropout_value}\n")
        f.write(f"Training Loss: {history.history['loss']}\n")
        f.write(f"Validation Loss: {history.history['val_loss']}\n")
        f.write(f"Training Accuracy: {history.history['accuracy']}\n")
        f.write(f"Validation Accuracy: {history.history['val_accuracy']}\n\n")

        # 绘制Loss曲线
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Dropout = {dropout_value}')
        plt.legend()
        plt.savefig(f'./picture/loss_plot_dropout_{dropout_value}.png')

        # 绘制Accuracy曲线
        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Dropout = {dropout_value}')
        plt.legend()
        plt.savefig(f'./picture/accuracy_plot_dropout_{dropout_value}.png')

# 选择最佳的Dropout参数
# 根据生成的曲线选择表现最好的模型，可以根据Validation Loss和Validation Accuracy来判断模型的性能
