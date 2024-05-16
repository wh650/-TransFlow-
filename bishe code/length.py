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

# 数据预处理
lengths_to_test = range(10, 100, 5)  # 截取长度范围为10到50，步长为5
results = []

for max_seq_length in lengths_to_test:
    data_chunks = pd.read_csv('selected_proteins300_400stuff.csv', chunksize=batch_size)  # 重新创建data_chunks

    tokenizer = Tokenizer(char_level=True)
    family_labels = []
    sequences = []

    for chunk in data_chunks:
        for index, row in chunk.iterrows():
            family_labels.append(row['family_label'])
            seq = row['protein_sequence'][:max_seq_length]
            seq = seq.ljust(max_seq_length, 'X')  # 使用字符'X'补齐至指定长度
            sequences.append(seq)

    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    sequences = pad_sequences(sequences, maxlen=max_seq_length)

    # 将家族标签转换为数字编码
    label_to_index = {label: index for index, label in enumerate(np.unique(family_labels))}
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

    # 构建模型和训练，评估模型的代码略去不变

    # 构建模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_seq_length),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_to_index), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    # 训练模型
    model.fit(X_train_tf, y_train_tf, validation_data=(X_val_tf, y_val_tf), epochs=10, batch_size=128)

    # 在测试集上评估模型
    loss, accuracy = model.evaluate(X_test_tf, y_test_tf)
    results.append({'Length': max_seq_length, 'Test Loss': loss, 'Test Accuracy': accuracy})
    print(f'截取每个蛋白质的前{max_seq_length}个氨基酸得到模型 Test loss: {loss} Test accuracy: {accuracy}')

# 将结果保存到CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv('length2.csv', index=False)

