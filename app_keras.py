import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# 1. Preparar os dados
X_train_balanced = np.array(X_train_balanced)  # Converter para arrays NumPy
y_train_balanced = np.array(y_train_balanced)
X_val = np.array(X_val)
y_val = np.array(y_val)

# 2. Criar o modelo de rede neural
model = Sequential()

# Camada de entrada e primeira camada escondida
model.add(Dense(128, input_dim=X_train_balanced.shape[1], activation='relu'))  # 128 neurônios
model.add(Dropout(0.3))  # Dropout para evitar overfitting

# Segunda camada escondida
model.add(Dense(64, activation='relu'))  # 64 neurônios
model.add(Dropout(0.3))

# Camada de saída
model.add(Dense(1, activation='sigmoid'))  # Saída binária (fraude ou não)

# 3. Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  # Perda para classificação binária
              metrics=['accuracy'])  # Métrica de avaliação

# 4. Treinar o modelo
history = model.fit(
    X_train_balanced, y_train_balanced,
    validation_data=(X_val, y_val),
    epochs=50,  # Número de épocas
    batch_size=32,  # Tamanho do lote
    verbose=1  # Exibir progresso
)

# 5. Avaliar o modelo
y_pred = (model.predict(X_val) > 0.5).astype(int)  # Predição com limiar 0.5
print("Relatório de Classificação:\n", classification_report(y_val, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_val, y_pred))

# 6. Visualizar o histórico de treinamento
import matplotlib.pyplot as plt

# Gráfico de perda
plt.plot(history.history['loss'], label='Perda - Treinamento')
plt.plot(history.history['val_loss'], label='Perda - Validação')
plt.legend()
plt.title("Evolução da Perda")
plt.xlabel("Épocas")
plt.ylabel("Perda")
plt.show()

# Gráfico de acurácia
plt.plot(history.history['accuracy'], label='Acurácia - Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia - Validação')
plt.legend()
plt.title("Evolução da Acurácia")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.show()
