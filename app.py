import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Configurar a página do Streamlit
st.set_page_config(page_title="Fraude em Transações", layout="wide")

# Título da Aplicação
st.title("Detecção de Fraude em Transações")

# Fluxo Geral
st.sidebar.header("Fluxo Geral")
steps = [
    "Carregamento e Exploração dos Dados",
    "Análise Exploratória",
    "Divisão do Dataset",
    "Treinamento de Modelos",
    "Avaliação de Modelos"
]
for step in steps:
    st.sidebar.write(f"✔️ {step}")

# Carregamento de Dados
st.header("1. Carregamento e Exploração dos Dados")
dataset_path = "C:\\Repositorios_GitHube\\MeusProjetos\\IA_IdentificacaoFraude\\data\\raw\\dataset_transacoes.csv"

df = pd.read_csv(dataset_path)

if st.checkbox("Visualizar Dados"):
    st.write(df.head())

if st.checkbox("Informações Gerais do Dataset"):
    st.write(df.info())

if st.checkbox("Resumo Estatístico"):
    st.write(df.describe())

# Análise Exploratória
st.header("2. Análise Exploratória")
if st.checkbox("Visualizar Distribuição das Classes"):
    fig, ax = plt.subplots()
    sns.countplot(x="classe", data=df, ax=ax)
    ax.set_title("Distribuição das Classes (Fraude vs Não Fraude)")
    st.pyplot(fig)

if st.checkbox("Matriz de Correlação"):
    categorical_cols = [
        "localizacao", "tipo_estabelecimento", "dia_semana",
        "categoria_despesa", "genero_titular", "historico_pagamento"
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de Correlação (Atributos Codificados)")
    st.pyplot(fig)

# Divisão do Dataset
st.header("3. Divisão do Dataset")
numerical_cols = [
    "valor_transacao", "limite_credito", "saldo_atual",
    "valor_medio_transacoes", "utilizacao_credito", "tempo_desde_ultima_transacao"
]

# Normalização
scaler = MinMaxScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

X = df_encoded.drop("classe", axis=1)
y = df_encoded["classe"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

if st.checkbox("Tamanhos dos Conjuntos"):
    st.write(f"Tamanho do conjunto de treino: {X_train.shape}")
    st.write(f"Tamanho do conjunto de validação: {X_val.shape}")
    st.write(f"Tamanho do conjunto de teste: {X_test.shape}")

# Treinamento e Avaliação de Modelos
st.header("4. Treinamento e Avaliação de Modelos")

# SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_balanced, y_train_balanced)

# Avaliação
y_pred = model_rf.predict(X_val)
if st.checkbox("Avaliação com Random Forest"):
    st.write("Relatório de Classificação:\n", classification_report(y_val, y_pred, output_dict=True))
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de Confusão")
    st.pyplot(fig)

# XGBoost
if st.checkbox("Treinar com XGBoost"):
    model_xgb = XGBClassifier(random_state=42, scale_pos_weight=len(y_train_balanced) / sum(y_train_balanced == 1))
    model_xgb.fit(X_train_balanced, y_train_balanced)
    y_pred_xgb = model_xgb.predict(X_val)

    st.write("Relatório de Classificação (XGBoost):\n", classification_report(y_val, y_pred_xgb, output_dict=True))
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_val, y_pred_xgb), annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de Confusão (XGBoost)")
    st.pyplot(fig)
