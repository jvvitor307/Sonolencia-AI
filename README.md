# Detecção de Fadiga em Motoristas com Visão Computacional

Sistema de detecção de fadiga em tempo real usando MediaPipe Face Mesh e EAR/MAR geométrico.

## Arquitetura

```
Camera → MediaPipe Face Mesh → EAR/MAR Geométrico → Alerta
             (468 pontos)      ( Razão de Aspecto )
```

## Estrutura do Projeto

```
.
├── main.py                    # Ponto de entrada
├── requirements.txt
├── notebooks/
│   └── analise_resultados.ipynb
├── src/
│   ├── config.py              # Configurações e hiperparâmetros
│   ├── landmarks.py           # Cálculo de EAR, MAR e detecção
│   └── detector.py            # Detecção em tempo real
├── data/
│   └── models/                # Modelo Face Landmarker (.task)
└── results/                   # Gráficos e relatórios
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py --camera 0
```

Controles:
- `q` — sair
- `g` — salvar gráfico EAR/MAR em tempo real

Opções adicionais:
```bash
python main.py --save --output video.avi  # Salvar vídeo
```

## Fundamentação Matemática

### EAR (Eye Aspect Ratio)

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

- $p_1$ = canto externo, $p_4$ = canto interno
- $p_2, p_3$ = pálpebra superior; $p_5, p_6$ = pálpebra inferior
- EAR < 0.21 por 1 segundo → olhos fechados

### MAR (Mouth Aspect Ratio)

$$MAR = \frac{||p_{13} - p_{14}||}{||p_{78} - p_{308}||}$$

- $p_{13}, p_{14}$ = lábio superior e inferior
- $p_{78}, p_{308}$ = cantos da boca
- MAR > 0.6 por 10 frames → bocejo detectado

## Métricas

O sistema gera automaticamente:

| Métrica | Arquivo |
|---------|---------|
| Gráfico EAR/MAR tempo real | `results/ear_mar_realtime.png` |
| Métricas de inferência | `results/inference_metrics.txt` |
