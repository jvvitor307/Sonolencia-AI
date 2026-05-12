# Detecção de Fadiga em Motoristas com Visão Computacional

Sistema de detecção de fadiga em tempo real usando MediaPipe Face Mesh, EAR/MAR geométrico e CNN (MobileNetV2).

## Arquitetura

```
Camera → MediaPipe Face Mesh → EAR/MAR Geométrico → CNN (validação) → Alerta
             (468 pontos)      ( Razão de Aspecto )   (MobileNetV2)
```

## Estrutura do Projeto

```
.
├── main.py                    # Ponto de entrada (CLI)
├── requirements.txt
├── notebooks/
│   └── analise_resultados.ipynb
├── src/
│   ├── config.py              # Configurações e hiperparâmetros
│   ├── data_preparation.py    # Download e organização dos datasets
│   ├── train_model.py         # Treinamento da CNN (MobileNetV2)
│   ├── landmarks.py           # Cálculo de EAR, MAR e detecção
│   ├── detector.py            # Detecção em tempo real
│   └── evaluate.py            # Avaliação e métricas
├── data/
│   ├── raw/                   # Dados brutos
│   ├── processed/             # Dados organizados (train/val)
│   └── models/                # Modelos treinados (.h5)
└── results/                   # Gráficos e relatórios
```

## Instalação

```bash
pip install -r requirements.txt
```

> Para GPUs NVIDIA: instale o CUDA toolkit e cuDNN compatíveis com TensorFlow.

## Uso — Pipeline Completo em 3 Passos

### Passo 1: Preparar o Dataset

```bash
python main.py prepare --dataset drowsiness
```

Baixa o Drowsiness Dataset do Kaggle e organiza em `data/processed/{train,val}/{Closed,Open,yawn,no_yawn}`.

### Passo 2: Treinar o Modelo

```bash
python main.py train --model mobilenetv2
```

Treina MobileNetV2 com transfer learning (backbone ImageNet) + fine-tuning.
O modelo é salvo em `data/models/eye_classifier_cnn.h5`.

### Passo 3: Executar Detecção

```bash
python main.py detect --camera 0
```

Controles:
- `q` — sair
- `g` — salvar gráfico EAR/MAR em tempo real

Opções adicionais:
```bash
python main.py detect --no-cnn          # Sem validação CNN (mais rápido)
python main.py detect --save --output video.avi  # Salvar vídeo
```

### Avaliar Modelo

```bash
python main.py evaluate --full
```

Gera: matriz de confusão, curvas ROC, F1-Score, benchmark de inferência, distribuição EAR.

## Fundamentação Matemática

### EAR (Eye Aspect Ratio)

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

- $p_1$ = canto externo, $p_4$ = canto interno
- $p_2, p_3$ = pálpebra superior; $p_5, p_6$ = pálpebra inferior
- EAR < 0.21 por 15 frames consecutivos → olhos fechados

### MAR (Mouth Aspect Ratio)

$$MAR = \frac{||p_{13} - p_{14}||}{||p_{78} - p_{308}||}$$

- $p_{13}, p_{14}$ = lábio superior e inferior
- $p_{78}, p_{308}$ = cantos da boca
- MAR > 0.6 por 10 frames → bocejo detectado

## Métricas para o Relatório

O sistema gera automaticamente:

| Métrica | Arquivo |
|---------|---------|
| Acurácia / F1-Score | `results/final_classification_report.txt` |
| Matriz de Confusão | `results/final_confusion_matrix.png` |
| Curvas ROC (AUC) | `results/roc_curves.png` |
| Gráfico EAR/MAR tempo real | `results/ear_mar_realtime.png` |
| Histórico de treino | `results/training_history.png` |
| Benchmark FPS | `results/inference_benchmark.txt` |
| Métricas de inferência | `results/inference_metrics.txt` |

## Datasets Utilizados

| Dataset | Conteúdo | Uso |
|---------|----------|-----|
| Drowsiness Dataset (Kaggle) | Olhos abertos/fechados + bocejos | Treino CNN principal |
| MRL Eye Dataset (Kaggle) | 37k imagens de olhos | Refinamento do classificador de olhos |
