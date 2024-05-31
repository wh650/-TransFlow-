import os
os.chdir('./data')

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# 读取数据
batch_size = 5000
data_chunks = pd.read_csv('selected_proteins5000_400stuff.csv', chunksize=batch_size)

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
epochs = 20

# 构建模型
input_layer = tf.keras.layers.Input(shape=(max_seq_length,))
embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_seq_length)(input_layer)
transformer_block = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=embedding_dim)(embedding, embedding)
transformer_block = tf.keras.layers.Dense(256, activation='relu')(transformer_block)
transformer_block = tf.keras.layers.Dropout(0.3)(transformer_block)
transformer_block = tf.keras.layers.LayerNormalization(epsilon=1e-6)(transformer_block)

lstm_output = tf.keras.layers.LSTM(256, return_sequences=True)(transformer_block)  # 将Transformer的输出作为LSTM的输入
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
model.fit(X_train_tf, y_train_tf, validation_data=(X_val_tf, y_val_tf), epochs=epochs, batch_size=batch_size)

# 模型评估
history = model.fit(X_train_tf, y_train_tf, validation_data=(X_val_tf, y_val_tf), epochs=epochs, batch_size=256)

# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test_tf, y_test_tf)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# 计算精确率、召回率、F1分数和混淆矩阵
y_probs = model.predict(X_test_tf)
y_pred = np.argmax(y_probs, axis=1)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

# 将结果保存到result文件夹中
result_dir = './result'
os.makedirs(result_dir, exist_ok=True)

np.savetxt(os.path.join(result_dir, 'confusion_matrix.txt'), conf_matrix, fmt='%d')

with open(os.path.join(result_dir, 'evaluation_metrics.txt'), 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1 Score: {f1}')

# 绘制ROC曲线和计算AUC值
y_probs = model.predict(X_test_tf)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(label_to_index)):
    fpr[i], tpr[i], _ = roc_curve(y_test_tf, y_probs[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算平均AUC值


plt.figure()
for i in range(len(label_to_index)):
    plt.plot(fpr[i], tpr[i], lw=2, label='Class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')

# 保存AUC值到CSV文件
auc_df = pd.DataFrame.from_dict(roc_auc, orient='index', columns=['AUC'])
auc_df.index.name = 'Class'
auc_df.to_csv(os.path.join(result_dir, 'auc_scores.csv'))

plt.savefig(os.path.join(result_dir, 'roc_curve.png'))
plt.close()



import seaborn as sns

# 绘制混淆矩阵热图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(result_dir, 'confusion_matrix_heatmap.png'))
plt.close()

# 绘制学习曲线
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig(os.path.join(result_dir, 'accuracy_learning_curve.png'))
plt.close()

# 绘制验证集和训练集准确率随训练周期变化的曲线
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig(os.path.join(result_dir, 'accuracy_learning_curve2.png'))
plt.close()
