import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# 1. Carregar e processar os dados
# Supondo que 'df_encoded' já foi preparado com todas as colunas categóricas codificadas
X = df_encoded.drop("classe", axis=1).values
y = df_encoded["classe"].values

# Divisão inicial dos dados
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Aplicar normalização
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Aplicar SMOTE para balancear as classes no conjunto de treino
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 2. Criar o modelo de rede neural
model = Sequential([
    Dense(64, input_dim=X_train_balanced.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Saída binária
])

# Compilar o modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Configurar Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 3. Treinar o modelo
history = model.fit(
    X_train_balanced, y_train_balanced,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 4. Avaliar o modelo
# Previsões no conjunto de validação
y_pred = (model.predict(X_val) > 0.5).astype(int)
roc_auc = roc_auc_score(y_val, y_pred)

# Exibir métricas
print("Relatório de Classificação:\n", classification_report(y_val, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_val, y_pred))
print(f"AUC-ROC: {roc_auc:.2f}")

# 5. Visualizar o histórico de treinamento
# Gráfico de perda
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Gráfico de acurácia
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
