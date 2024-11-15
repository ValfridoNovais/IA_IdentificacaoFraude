# Importação das bibliotecas necessárias
import os  # Trabalhar com sistema de arquivos
import pandas as pd
import random
import numpy as np

# Definição do número de linhas do dataset
num_samples = 10000

def gerar_numero_cartao():
    """
    Função para gerar um número completo de cartão de crédito válido.
    Segue a estrutura:
    - Primeiro dígito: tipo de instituição (4 para Visa, por exemplo).
    - Próximos 5 dígitos: identificador do banco.
    - Próximos 9 dígitos: número exclusivo do cliente.
    - Último dígito: dígito verificador calculado pelo algoritmo de Luhn.
    """
    # Primeiro dígito: Identificador de indústria (4 para Visa neste exemplo)
    primeiro_digito = "4"
    
    # Próximos 5 dígitos: Identificador do emissor (BIN)
    identificador_emissor = f"{random.randint(10000, 99999)}"
    
    # Próximos 9 dígitos: Número exclusivo do cliente
    numero_cliente = f"{random.randint(100000000, 999999999)}"
    
    # Número parcial sem o dígito verificador
    numero_parcial = primeiro_digito + identificador_emissor + numero_cliente
    
    # Cálculo do dígito verificador pelo algoritmo de Luhn
    def calcular_digito_luhn(numero):
        soma = 0
        alternar = False
        for digito in reversed(numero):
            n = int(digito)
            if alternar:
                n *= 2
                if n > 9:
                    n -= 9
            soma += n
            alternar = not alternar
        return (10 - (soma % 10)) % 10
    
    digito_verificador = calcular_digito_luhn(numero_parcial)
    
    # Retorna o número completo do cartão
    return numero_parcial + str(digito_verificador)

# Geração de números de cartão para o dataset
data = {
    "numero_cartao": [gerar_numero_cartao() for _ in range(num_samples)],  # Chama a função para cada entrada
    "valor_transacao": np.random.uniform(1, 5000, num_samples),
    "localizacao": random.choices(["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba"], k=num_samples),
    "tipo_estabelecimento": random.choices(["Restaurante", "E-commerce", "Supermercado", "Posto de Gasolina"], k=num_samples),
    "hora_transacao": np.random.randint(0, 24, num_samples),
    "dia_semana": random.choices(["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"], k=num_samples),
    "categoria_despesa": random.choices(["Alimentação", "Transporte", "Lazer", "Saúde"], k=num_samples),
    "idade_titular": np.random.randint(18, 80, num_samples),
    "genero_titular": random.choices(["Masculino", "Feminino", "Outro"], k=num_samples),
    "historico_pagamento": random.choices(["Bom", "Atrasado", "Inadimplente"], k=num_samples),
    "limite_credito": np.random.uniform(1000, 50000, num_samples),
    "saldo_atual": np.random.uniform(0, 50000, num_samples),
    "num_transacoes_anteriores": np.random.randint(0, 100, num_samples),
    "num_cartoes_suplementares": np.random.randint(0, 5, num_samples),
    "valor_medio_transacoes": np.random.uniform(10, 2000, num_samples),
    "num_transacoes_outros_paises": np.random.randint(0, 20, num_samples),
    "utilizacao_credito": np.random.uniform(0, 1, num_samples),
    "transacoes_suspeitas": np.random.randint(0, 10, num_samples),
    "tempo_desde_ultima_transacao": np.random.uniform(0, 30, num_samples),
    "classe": random.choices([0, 1], weights=[85, 15], k=num_samples),
}

# Criação do DataFrame
df = pd.DataFrame(data)

# Caminho para salvar o arquivo
output_dir = os.path.join("data", "raw")
output_file = os.path.join(output_dir, "dataset_transacoes.csv")

# Garantir que os diretórios existem
if not os.path.exists(output_dir):
    print(f"Criando diretório: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

# Salvando o dataset em um arquivo CSV
df.to_csv(output_file, index=False)
print(f"Dataset salvo em: {output_file}")
