import streamlit as st

# Incluir explicación y ecuaciones en Markdown con soporte para LaTeX


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import streamlit as st

# Definir funciones necesarias
def gini(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    return 2 * roc_auc - 1

def plot_gini_curve(df):
    # Ordenar por probabilidades de default en orden descendente
    df = df.sort_values(by='default_prob', ascending=False).reset_index(drop=True)

    # Calcular la curva de Lorenz para el Gini
    total_defaults = df['default'].sum()
    df['cum_defaults'] = df['default'].cumsum() / total_defaults
    df['cum_population'] = (np.arange(len(df)) + 1) / len(df)

    plt.figure()
    plt.plot(df['cum_population'], df['cum_defaults'], label=f'Curva de Gini (Índice de Gini = {gini(df["default"], df["default_prob"]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Fracción acumulada de la población')
    plt.ylabel('Fracción acumulada de defaults')
    plt.title('Curva de Gini')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Simular datos de RSE
np.random.seed(42)
n_samples = 10000

# Función para actualizar el gráfico, la matriz de confusión, y el Gini basado en los parámetros
def update_plot(k_RSE, c_RSE, RSE_max):
    # Regenerar los valores de RSE con el rango máximo ajustable
    debt_mean = 1.0
    debt_std = 0.5
    RSE = np.clip(np.random.normal(debt_mean, debt_std, n_samples), 0, RSE_max)

    # Calcular las probabilidades de default usando la función logística
    default_probs = 1 / (1 + np.exp(-(k_RSE * (RSE - c_RSE))))

    # Generar los defaults utilizando la función binomial
    defaults = np.random.binomial(1, default_probs)

    df = pd.DataFrame({
        'RSE': RSE,
        'default_prob': default_probs,
        'default': defaults
    })

    gini_value = gini(df['default'], df['default_prob'])

    # Umbral de decisión para predicciones
    threshold = 0.5
    predicted_defaults = (df['default_prob'] >= threshold).astype(int)

    # Matriz de confusión
    cm = confusion_matrix(df['default'], predicted_defaults)
    tn, fp, fn, tp = cm.ravel()

    # Mostrar matriz de confusión como tabla
    st.write("Matriz de Confusión:")
    cm_df = pd.DataFrame({
        'Predicho No Default': [tn, fn],
        'Predicho Default': [fp, tp]
    }, index=['Real No Default', 'Real Default'])

    st.table(cm_df)

    # Mostrar estadísticas descriptivas
    num_defaults = df['default'].sum()
    num_non_defaults = len(df) - num_defaults
    st.write(f"Número de Defaults 'Reales': {num_defaults}")
    st.write(f"Número de No Defaults 'Reales': {num_non_defaults}")

    # Graficar curva de densidad de RSE para No Default y Default
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df[df['default'] == 0]['RSE'], label='Sin Default', color='blue', fill=True, alpha=0.5)
    sns.kdeplot(df[df['default'] == 1]['RSE'], label='Con Default', color='red', fill=True, alpha=0.5)

    plt.xlabel('Ratio Deuda/Ingreso (RSE)')
    plt.ylabel('Densidad')
    plt.legend(loc='upper right')
    plt.title(f'Curva de Densidad de RSE para Default y Sin Default\nGini: {gini_value:.4f}')
    st.pyplot(plt)

    # Graficar curva Gini
    plot_gini_curve(df)

# Crear sliders interactivos
k_RSE = st.sidebar.slider('Pendiente (k_RSE)', 0.01, 20.0, 1.0, 0.1)
c_RSE = st.sidebar.slider('Desplazamiento (c_RSE)', 0.0, 5.0, 1.5, 0.1)
RSE_max = st.sidebar.slider('Máximo RSE', 1.0, 10.0, 3.0, 0.1)

# Actualizar gráficos basados en los valores de los sliders
update_plot(k_RSE, c_RSE, RSE_max)


st.markdown(r"""
# Calculadora de Gini sintético con supuestos algo heroicos 

Se usan ecuaciones para calcular las probabilidades de default, generar los defaults, y calcular el Gini. Por simplicidad, se usa una única variable sintética, "ratio de sobre-endeudamiento":

1. **Ecuación para el Ratio de sobre endeudamiento (RSE):**

El **Ratio de Sobreendeudamiento (RSE)** se genera utilizando una distribución normal truncada (truncada porque se asume que no puede tener valores de apalancamiento negativos o demasiado altos):

$$
RSE = \max\left(0, \min\left(RSE_{\text{max}}, \mathcal{N}(\mu_{\text{deuda}}, \sigma_{\text{deuda}})\right)\right)
$$

Donde:
- $\mu_{\text{deuda}}$ = 1.0: Media del RSE.
- $\sigma_{\text{deuda}}$ = 0.5: Desviación estándar del RSE.
- $\mathcal{N}$ representa la distribución normal.
- $RSE_{\text{max}}$ es un parámetro ajustable que determina el valor máximo del RSE.

2. **Función Logística para Calcular la Probabilidad de Default:**

La probabilidad de default $P(\text{default})$ se calcula utilizando la siguiente ecuación logística, incorporando en "z" el proceso generador de datos del RSE:

$$
P(\text{default} \mid RSE) = \frac{1}{1 + e^{-z}}
$$

Donde:
$$
z = k_{\text{RSE}} \cdot (RSE - c_{\text{RSE}})
$$

3. **Generación de Defaults Usando la Distribución Binomial:**

Después de calcular las probabilidades de default, se generan los eventos de default utilizando una distribución binomial:

$$
\text{default} = \text{Binomial}(1, P(\text{default}))
$$

4. **Cálculo del Gini:**

El **Gini** se calcula a partir de la curva ROC, que compara la tasa de verdaderos positivos (True Positive Ratio - TPR) con la tasa de falsos positivos (False Positive Ratio - FPR):

$$
Gini = 2 \times \text{AUC} - 1
$$

Donde:
- **AUC**: Área bajo la curva ROC, que se obtiene al trazar la TPR respecto a la FPR.

5. **Matriz de Confusión:**

Con el fin de observar la performance del modelo, usamos la matriz de confusión en sus 4 cuadrantes:

- **Verdaderos Positivos (TP)**: Casos correctamente clasificados como defaults.
- **Verdaderos Negativos (TN)**: Casos correctamente clasificados como no defaults.
- **Falsos Positivos (FP)**: Casos incorrectamente clasificados como defaults.
- **Falsos Negativos (FN)**: Casos incorrectamente clasificados como no defaults.

La matriz de confusión se calcula con:

$$
\begin{array}{|c|c|}
\hline
\text{Predicción} & \text{Observación} \\
\hline
\text{TP} & \text{FN} \\
\hline
\text{FP} & \text{TN} \\
\hline
\end{array}
$$
""")
