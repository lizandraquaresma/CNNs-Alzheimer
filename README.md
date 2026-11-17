# Classificação de Demência por Imagens de Ressonância Magnética

Este projeto usa redes neurais convolucionais (CNNs) para classificar imagens de ressonância magnética conforme o estágio de demência.

O notebook [`MRIReport.ipynb`](MRIReport.ipynb) registra uma rodada de classificação em três estágios de demência.

## Publicação

Rafael Farias Batista, Bianka dos Santos Gouvêa, Lizandra de Lima Quaresma e Marta de Oliveira Barreiros (2025). *Classificação de estágios da doença de Alzheimer em imagens de ressonância magnética por redes neurais convolucionais*. Revista Eletrônica Acervo Saúde, 25(8), e21133. [DOI do artigo](https://doi.org/10.25248/reas.e21133.2025)

## Objetivo

Avaliar modelos de aprendizado profundo na classificação de imagens de MRI nas seguintes classes:

- `Very mild Dementia` — demência muito leve
- `Mild Dementia` — demência leve
- `Moderate Dementia` — demência moderada

O código contém alternativas comentadas para incluir `Non Demented` ou agrupar os estágios em `Dementia`. As saídas salvas correspondem ao cenário ativo de três classes.

## Arquiteturas avaliadas

- VGG19
- ResNet50
- InceptionV3
- EfficientNetB7
- DenseNet201
- CNN personalizada

VGG19, InceptionV3, EfficientNetB7 e DenseNet201 usam bases ImageNet congeladas no código disponível. A ResNet50 também tem a base congelada, com `weights=None` na configuração atual. As chamadas de treino e avaliação da AlexNet estão comentadas.

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

As imagens usadas no notebook vieram do *Well-Documented Alzheimer's Dataset*, uma coleção do Kaggle derivada do OASIS-1. A descrição fornecida para a versão do Kaggle informa recorte e redimensionamento para 224 × 224 com preenchimento, aumento de dados via Roboflow e uma divisão de origem por participante. Ela credita a Yiwei Lu a conversão e a remoção de crânio e de outros tecidos com LinkNet3D. O notebook recebe imagens já preparadas, converte-as para RGB e as redimensiona. [Coleção Kaggle](https://doi.org/10.34740/kaggle/dsv/10215637) · [OASIS-1](https://doi.org/10.1162/jocn.2007.19.9.1498)

Os dados de origem foram fornecidos pelo OASIS-1: Cross-Sectional (D. Marcus, R. Buckner, J. Csernansky e J. Morris), com apoio de P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616 e U24 RR021382. [Créditos do OASIS](https://bpb-us-w2.wpmucdn.com/sites.wustl.edu/dist/6/4383/files/2025/07/Data-Use-Agreement_July2025.pdf)

O notebook seleciona arquivos das pastas de classe em `DATA_DIR` e cria novas partições por imagem. O lote principal tem 450 imagens por classe e é dividido em treino, validação e teste interno. A avaliação salva usa um lote adicional de 25 imagens por classe, também amostrado de `DATA_DIR`. A divisão por participante descrita pelo Kaggle, por si só, não comprova que essas novas partições sejam independentes. Sem o manifesto histórico de imagens e participantes, a sobreposição permanece sem verificação.

Os dados não são distribuídos neste repositório. Consulte os termos da fonte de obtenção e os termos aplicáveis do OASIS antes de compartilhar imagens, manifestos ou derivados. [Termo de uso do OASIS](https://bpb-us-w2.wpmucdn.com/sites.wustl.edu/dist/6/4383/files/2025/07/Data-Use-Agreement_July2025.pdf)

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

- seleção das imagens com semente fixa e dependente da ordem dos arquivos enumerados;
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

Os melhores resultados agregados registrados no lote adicional de 75 imagens foram:

| Modelo | Acurácia registrada |
|---|---:|
| ResNet50 | 96,00% |
| CNN personalizada | 94,67% |

As acurácias arredondadas dos cinco modelos em comum coincidem com a Tabela 6 do artigo, referente à avaliação dos três estágios de demência. Os resultados do cenário de quatro classes e do cenário binário apresentados no artigo não aparecem nas saídas salvas deste notebook. A metodologia publicada também descreve uma divisão de 76/19/5 e Adam com taxa de `1e-3`. O código disponível cria um teste interno além do lote adicional e usa `1e-4` nas cinco arquiteturas Keras Applications. Essas diferenças limitam a atribuição das saídas a uma configuração histórica específica. [Artigo, metodologia e Tabela 6](https://acervomais.com.br/saude/pt_BR/article/download/21133/11433/244646)

**Nota sobre os resultados por classe.** Na versão anterior do código, os nomes das linhas do relatório seguiam a lista de pastas, em ordem diferente da codificação de classes. A versão atual usa a ordem do codificador nas novas avaliações, inclusive na matriz de confusão. As saídas históricas foram preservadas. A atribuição das métricas às classes nessas saídas e na Tabela 6 requer confirmação da ordem usada na execução original. A divergência de nomes não afeta a acurácia agregada.

Esses valores dependem do conjunto de dados, da amostragem, do particionamento e da versão das bibliotecas utilizadas. Recomenda-se não interpretar os valores como validação clínica ou métrica definitiva de desempenho.

## Limitações

- Os resultados dependem da qualidade e da representatividade do conjunto de MRI.
- Este projeto é experimental e não fornece diagnóstico médico.
- O modelo não substitui avaliação de um profissional de saúde.
- Os arquivos de pesos treinados não estão incluídos no repositório.
- O notebook depende de uma estrutura específica de diretórios no Google Drive.

## Licença

O código e a documentação do projeto são disponibilizados sob a licença MIT. Os dados e a publicação seguem seus próprios termos. Consulte o arquivo [`LICENSE`](LICENSE).

## Referência de execução

```text
Notebook: MRIReport.ipynb
Ambiente recomendado: Google Colaboratory com GPU
```
