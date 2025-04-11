# Análise de Volatilidade Cambial - 2024

Este projeto apresenta uma análise de volatilidade cambial utilizando modelos GARCH (Generalized Autoregressive Conditional Heteroskedasticity) aplicados às variações percentuais mensais das principais moedas do mundo em relação ao Real (BRL) durante o ano de 2024.

## 📈 Objetivo
Avaliar o comportamento da volatilidade das taxas de câmbio para entender os riscos associados a essas moedas e como flutuações podem impactar o cenário macroeconômico e decisões financeiras.

## 🛠️ Tecnologias Utilizadas
- Python
- Pandas
- Matplotlib
- Statsmodels
- ARCH (Modelos GARCH)

## 📊 Moedas Analisadas
- USD/BRL (Dólar Americano)
- EUR/BRL (Euro)
- GBP/BRL (Libra Esterlina)
- ARS/BRL (Peso Argentino)
- ZAR/BRL (Rand Sul-Africano)

## 🔍 Metodologia
1. Simulação de variações mensais com base em padrões históricos.
2. Aplicação do modelo GARCH(1,1) para estimar a volatilidade condicional.
3. Visualização gráfica dos resultados para análise comparativa.

## 📎 Execução
Antes de executar, instale as dependências:

```bash
pip install pandas matplotlib arch

