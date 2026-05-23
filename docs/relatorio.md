# Relatório de Resultados — Sonolencia-AI

**Autores:** João Vitor Melo Fontenele, Caio Vinicius Carvalho Rocha, José Lucas Vasconcelos de Lucena
**Disciplina:** Visão Computacional — Pós-graduação
**Data:** Maio de 2026

---

## 1. Descrição da Aplicação

O **Sonolencia-AI** é um sistema de detecção de fadiga em motoristas que opera em tempo real via webcam. A aplicação captura o vídeo da câmera, detecta o rosto do motorista usando o MediaPipe Face Mesh (468 pontos faciais) e calcula duas métricas geométricas:

- **EAR (Eye Aspect Ratio):** mede a abertura dos olhos. Valor baixo = olhos fechados.
- **MAR (Mouth Aspect Ratio):** mede a abertura da boca. Valor alto = bocejo.

Quando os olhos permanecem fechados por 1 segundo ou a boca fica aberta por 10 frames consecutivos, o sistema dispara um alerta visual na tela.

---

## 2. Como Executar

```bash
# Instalar dependências
uv sync

# Rodar o sistema
python main.py --camera 0

# Salvar vídeo
python main.py --camera 0 --save --output video.avi
```

Controles: `q` para sair, `g` para salvar gráfico EAR/MAR.

---

## 3. Arquitetura

```
main.py                  # Entrada CLI (argumentos: --camera, --save, --output)
src/
├── config.py            # Hiperparâmetros e índices dos landmarks
├── landmarks.py         # MediaPipe Face Mesh, cálculo de EAR/MAR, lógica de detecção
└── detector.py          # Loop principal, calibração, HUD, gráficos, relatórios
data/models/
└── face_landmarker.task # Modelo do MediaPipe
results/                 # Saídas geradas (gráficos, métricas)
```

---

## 4. Parâmetros do Sistema

| Parâmetro | Valor | Função |
|-----------|-------|--------|
| Limiar EAR | 0.21 | Abaixo disso = olhos fechados |
| Limiar MAR | 0.6 | Acima disso = boca aberta |
| Tempo olhos fechados | 1s | Para disparar alerta de sono |
| Frames consecutivos boca | 10 | Para confirmar bocejo |
| Duração calibração | 3.0s | Fase inicial de ajuste |
| Razão de calibração | 0.6 | Multiplicador sobre EAR médio do motorista |

---

## 5. Funcionalidades

- **Detecção de sonolência:** olhos fechados por ≥ 1s → alerta "FADIGA: DORMINDO"
- **Detecção de bocejo:** MAR > 0.6 por ≥ 10 frames → alerta "FADIGA: BOCEJO DETECTADO"
- **Calibração adaptativa:** nos primeiros 3 segundos, coleta o EAR do motorista e ajusta o limiar automaticamente
- **HUD visual:** FPS, valores de EAR/MAR, barras de progresso, overlay vermelho ao detectar fadiga
- **Gravação de vídeo:** opcional, formato AVI (XVID)
- **Gráfico em tempo real:** salva `results/ear_mar_realtime.png` com série temporal de EAR e MAR
- **Relatório de métricas:** salva `results/inference_metrics.txt` com estatísticas da sessão

---

## 6. Resultados Obtidos

Os resultados abaixo foram coletados em uma sessão de teste com webcam (640×480), processando 227 frames:

| Métrica | Valor |
|---------|-------|
| FPS médio | 30.25 |
| Tempo de inferência | 33.06 ms/frame |
| Limiar EAR calibrado | 0.218 |
| Frames processados | 227 |

| | Mínimo | Máximo | Média |
|---|--------|--------|-------|
| **EAR** | 0.047 | 0.392 | 0.308 |
| **MAR** | 0.007 | 0.145 | 0.039 |

- O EAR médio de 0.308 indica olhos abertos. O mínimo de 0.047 confirma detecção clara de fechamento ocular.
- O MAR permaneceu baixo (média 0.039), sem bocejos durante a sessão.
- A calibração ajustou o limiar EAR de 0.21 para 0.218, personalizando a detecção ao motorista.

---

## 7. Observações e Limitações

- Requer iluminação adequada; baixa luz prejudica a detecção facial
- Óculos escuros obstruem os landmarks oculares, inviabilizando o cálculo do EAR
- Detecta apenas um rosto por vez
- Não requer GPU — roda inteiramente em CPU a ~30 FPS
- A calibração adaptativa reduz falsos positivos para diferentes tipos faciais

---

## 8. Trabalhos Futuros

- Câmera infravermelha para operação noturna
- Alerta sonoro e vibração do volante
- Testes em condições reais de estrada
- Pré-processamento com CLAHE para iluminação adversa
- Suporte a múltiplos rostos
