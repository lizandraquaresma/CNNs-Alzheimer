# Classificação de Demência por Imagens de Ressonância Magnética

Este projeto usa Redes Neurais Convolucionais (CNNs) e Transfer Learning para classificar exames de ressonância magnética conforme o estágio de demência.

O notebook [`MRIReport.ipynb`](MRIReport.ipynb) compara diferentes arquiteturas pré-treinadas e uma CNN desenvolvida para o projeto.

## Objetivo

Avaliar modelos de aprendizado profundo na classificação de imagens de MRI nas seguintes classes:

- `Very mild Dementia` — demência muito leve
- `Mild Dementia` — demência leve
- `Moderate Dementia` — demência moderada

O código contém classes alternativas que podem ser ativadas para testar outros cenários, como `Non Demented` ou uma classificação binária entre `Non Demented` e `Dementia`.

## Arquiteturas avaliadas

- VGG19
- ResNet50
- InceptionV3
- EfficientNetB7
- DenseNet201
- AlexNet
- CNN personalizada

Os modelos pré-treinados usam pesos do ImageNet com as camadas superiores retiradas. A saída é adaptada ao número de classes do projeto por meio de uma cabeça composta por Pooling Global, Batch Normalization, camadas densas, Dropout e uma camada Softmax.

## Requisitos

O notebook foi desenvolvido para o Google Colaboratory e utiliza:

- Python
- TensorFlow/Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Pillow
- scikit-learn
- tqdm

Exemplo de instalação no ambiente do notebook:

```bash
pip install numpy pandas tensorflow opencv-python matplotlib pillow scikit-learn tqdm
```

## Organização do conjunto de dados

Monte o Google Drive e organize o conjunto em diretórios por classe:

```text
/Data/
├── Very mild Dementia/
│   ├── imagem-1.jpg
│   └── ...
├── Mild Dementia/
│   ├── imagem-1.jpg
│   └── ...
└── Moderate Dementia/
    ├── imagem-1.jpg
    └── ...
```

Por padrão, o notebook procura os dados em:

```text
/content/drive/MyDrive/UEPA/TCC/Data
```

Antes da execução, ajuste a variável `DATA_DIR` e, se necessário, a lista `CLASSES` no notebook.

## Como executar

1. Abra [`MRIReport.ipynb`](MRIReport.ipynb) no Google Colaboratory.
2. Habilite um ambiente de execução com GPU.
3. Monte o Google Drive quando solicitado.
4. Atualize o caminho do conjunto de dados em `DATA_DIR`.
5. Execute as células em ordem.

O notebook realiza:

- seleção reprodutível das imagens;
- conversão das imagens para RGB e redimensionamento;
- codificação one-hot dos rótulos;
- divisão entre treino, validação e teste;
- balanceamento dos pesos das classes;
- treinamento dos modelos;
- geração de curvas de acurácia e erro;
- avaliação por acurácia, matriz de confusão e relatório de classificação.

Durante o treinamento são utilizados:

- `EarlyStopping`;
- `ModelCheckpoint`;
- `ReduceLROnPlateau`;
- pesos balanceados por classe;
- validação com dados separados do treinamento.

## Resultados

Os melhores resultados observados no conjunto adicional de testes descrito no notebook foram:

| Modelo | Acurácia registrada |
|---|---:|
| ResNet50 | 96,00% |
| CNN personalizada | 94,67% |

Esses valores dependem do conjunto de dados, da amostragem, do particionamento e da versão das bibliotecas utilizadas. Recomenda-se não interpretar os valores como validação clínica ou métrica definitiva de desempenho.

## Limitações

- Os resultados dependem da qualidade e da representatividade do conjunto de MRI.
- Este projeto é experimental e não fornece diagnóstico médico.
- O modelo não substitui avaliação de um profissional de saúde.
- Os arquivos de pesos treinados não estão incluídos no repositório.
- O notebook depende de uma estrutura específica de diretórios no Google Drive.

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo [`LICENSE`](LICENSE).

## Referência de execução

```text
Notebook: MRIReport.ipynb
Ambiente recomendado: Google Colaboratory com GPU
```
