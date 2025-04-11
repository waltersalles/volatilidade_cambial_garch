!pip install arch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Simulação de dados: variação percentual mensal de câmbio (2024)
data = {
    "Data": pd.date_range(start="2024-01-01", periods=12, freq='M'),
    "USD/BRL": [0.8, -1.0, 0.5, -0.3, 0.2, -0.1, 0.4, 0.3, -0.6, 0.9, -0.4, 0.7],
    "EUR/BRL": [0.6, -0.7, 0.3, -0.2, 0.1, -0.05, 0.2, 0.1, -0.3, 0.5, -0.2, 0.4],
    "GBP/BRL": [1.0, -1.2, 0.6, -0.4, 0.3, -0.2, 0.5, 0.4, -0.7, 1.1, -0.5, 0.8],
    "ARS/BRL": [5.0, -4.5, 4.2, -3.8, 4.0, -3.5, 4.3, 4.1, -4.6, 5.2, -4.2, 4.8],
    "ZAR/BRL": [1.2, -1.1, 1.0, -0.9, 1.1, -0.8, 1.3, 1.2, -1.0, 1.4, -1.1, 1.5]
}
df = pd.DataFrame(data).set_index("Data")

# Função para análise GARCH
def analyze_volatility(series, title):
    model = arch_model(series, vol='Garch', p=1, q=1)
    res = model.fit(disp="off")
    fig = res.plot(annualize='D')
    plt.suptitle(f'Volatilidade Condicional - {title}', fontsize=14)
    plt.tight_layout()
    return res

# Executar para cada moeda
results = {}
for moeda in df.columns:
    results[moeda] = analyze_volatility(df[moeda], moeda)

plt.show()
