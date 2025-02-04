<h1>Classificando músicas do Spotify</h1>

<h1>Visão Geral do Projeto</h1>

Este projeto tem como objetivo a criação de um pipeline de Data Science para analisar dados musicais e classificá-los em categorias com base em suas características. A coluna de interesse é a "valence", que descreve a positividade transmitida por uma faixa. A análise tem foco na identificação de músicas agitadas e lentas.

<h1>Estrutura do Projeto</h1>

Leitura e exploração de dados: Importação de dados com pandas e análise preliminar das colunas e dimensões do conjunto de dados.

Visualização: Uso de matplotlib e seaborn para criar histogramas e matrizes de correlação.

Criação da Variável Target: Categorizamos a coluna valence em duas classes (agitada e lenta).

Codificação de dados categóricos: Uso de LabelEncoder para transformar dados textuais em valores numéricos.

Análise de Correlação: Avaliação das relações entre as variáveis musicais.

Balanceamento do Target: Análise da distribuição das classes alvo.

Split de Dados: Divisão em dados de treino e teste para futura modelagem.

<h1>Tecnologias Utilizadas</h1>

Python 3.x

Pandas

Matplotlib

Seaborn

Scikit-learn

<h1>Etapas do Projeto</h1>

1. Leitura e Exploração dos Dados

```python
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
df = pd.read_csv('dataset.csv')
print(df.shape)
df.head()
```

2. Visualização Inicial da Distribuição de Valence

```python
plt.hist(df['valence'], bins=20, color='blue', edgecolor='black')
plt.xlabel('Valence')
plt.ylabel('Frequência')
plt.title('Histograma da Coluna Valence')
plt.show()
```


3. Criação da Variável Target

A categorização da coluna valence foi feita com a seguinte função:

```python
def categorizar_valence(row):
    if row['valence'] > 0.5:
        return 'agitada'
    else:
        return 'lenta'

# Cria a nova coluna "target"
df['target'] = df.apply(categorizar_valence, axis=1)
df.head()
```

4. Seleção de Features para o Modelo

Remoção de colunas desnecessárias:

```python
df_musica = df.drop(['Unnamed: 0', 'track_id'], axis=1)
df_musica.head()
```

5. Codificação de Dados Categóricos

Transformamos colunas textuais em valores numéricos para serem utilizadas nos algoritmos de machine learning:

```python
from sklearn.preprocessing import LabelEncoder

# Função para codificação
def label_encoder_dataframe(df, columns_to_encode):
    le = LabelEncoder()
    for column in columns_to_encode:
        if column in df.columns:
            df[column] = le.fit_transform(df[column])
        else:
            print('A lista possui colunas que não existem no dataframe.')
    return df

colunas_a_codificar = ['artists', 'album_name', 'track_name', 'explicit', 'track_genre', 'target']
df_musica = label_encoder_dataframe(df_musica, colunas_a_codificar)
df_musica.head()
```

6. Análise de Correlação

A matriz de correlação ajuda a visualizar a relação entre diferentes atributos musicais:

```python
import seaborn as sns

correlation_matrix = df_musica.corr().round(2)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax, cmap='coolwarm')
plt.show()
```

7. Distribuição do Target

```python
print(round(df_musica['target'].value_counts(normalize=True) * 100, 2))
```

8. Separando Dados para Treino e Teste

Para preparação futura do modelo preditivo:

```python
from sklearn.model_selection import train_test_split

X = df_musica.drop('target', axis=1)
y = df_musica['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<h1>Conclusão</h1>

Este projeto fornece um exemplo prático de como criar um pipeline de Data Science do zero, desde a leitura e análise dos dados até a preparação para modelagem. Com isso, é possível demonstrar habilidades importantes para seu portfólio, incluindo exploração de dados, feature engineering e visualização.

<h1>Como Executar o Projeto - Local</h1>

Clone este repositório.

Certifique-se de ter o Python 3.x instalado.

Instale as dependências com pip install -r requirements.txt.

Execute os códigos apresentados neste README para reproduzir as análises.

<h1>Referências</h1>

Por Ana Raquel


Link da vídeo aula: https://www.youtube.com/watch?v=hV3ORe7F8Q4

Espero que você se divirta explorando e aprendendo com este projeto!
