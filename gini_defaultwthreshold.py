import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score
import seaborn as sns
import streamlit as st

# Definir funciones necesarias
def gini(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    return 2 * roc_auc - 1

def plot_gini_curve(df, predicted_defaults):
    # Ordenar por predicciones binarizadas en orden descendente
    df = df.sort_values(by='default_prob', ascending=False).reset_index(drop=True)

    # Calcular la curva de Lorenz para el Gini usando predicciones binarizadas
    total_defaults = df['default'].sum()
    df['cum_defaults'] = df['default'].cumsum() / total_defaults
    df['cum_population'] = (np.arange(len(df)) + 1) / len(df)

    plt.figure()
    plt.plot(df['cum_population'], df['cum_defaults'], label=f'Curva de Gini (Índice de Gini = {gini(df["default"], predicted_defaults):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Fracción acumulada de la población')
    plt.ylabel('Fracción acumulada de defaults')
    plt.title('Curva de Gini')
    plt.legend(loc="lower right")
    st.pyplot(plt)

def plot_density_by_threshold(df, threshold):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df['default_prob'], label='Probabilidades de Default', color='blue', fill=True, alpha=0.5)
    sns.kdeplot(df[df['default_prob'] >= threshold]['default_prob'], label='Clasificados como Default', color='red', fill=True, alpha=0.5)
    plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold = {threshold}')
    plt.xlabel('Probabilidad de Default')
    plt.ylabel('Densidad')
    plt.legend(loc='upper right')
    plt.title('Distribución de Probabilidades de Default y Threshold Aplicado')
    st.pyplot(plt)

# Simular datos de DTI
np.random.seed(42)
n_samples = 10000

# Función para actualizar el gráfico, la matriz de confusión, y el Gini basado en los parámetros
def update_plot(k_dti, c_dti, dti_max, threshold):
    # Regenerar los valores de DTI con el rango máximo ajustable
    debt_mean = 1.0
    debt_std = 0.5
    dti = np.clip(np.random.normal(debt_mean, debt_std, n_samples), 0, dti_max)

    # Calcular las probabilidades de default usando la función logística
    default_probs = 1 / (1 + np.exp(-(k_dti * (dti - c_dti))))

    # Generar los defaults utilizando la función binomial
    defaults = np.random.binomial(1, default_probs)

    df = pd.DataFrame({
        'dti': dti,
        'default_prob': default_probs,
        'default': defaults
    })

    # Utilizar el threshold ajustable para predicciones
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

    # Mostrar estadísticas de rendimiento del modelo
    precision = precision_score(df['default'], predicted_defaults)
    recall = recall_score(df['default'], predicted_defaults)
    st.write(f"Precisión: {precision:.2f}")
    st.write(f"Sensibilidad (Recall): {recall:.2f}")

    # Recalcular el Gini utilizando las predicciones binarizadas en lugar de las probabilidades continuas
    gini_value = gini(df['default'], predicted_defaults)
    st.write(f"Gini basado en el punto de corte: {gini_value:.4f}")

    # Graficar densidades de las probabilidades de default y los predichos como default según el threshold
    plot_density_by_threshold(df, threshold)

    # Graficar curva Gini basada en predicciones binarizadas
    plot_gini_curve(df, predicted_defaults)

# Crear sliders interactivos
k_dti = st.sidebar.slider('Pendiente (k_DTI)', 0.01, 20.0, 1.0, 0.1)
c_dti = st.sidebar.slider('Desplazamiento (c_DTI)', 0.0, 5.0, 1.5, 0.1)
dti_max = st.sidebar.slider('Máximo DTI', 1.0, 10.0, 3.0, 0.1)
threshold = st.sidebar.slider('Threshold (Punto de Corte)', 0.0, 1.0, 0.5, 0.01)

# Actualizar gráficos basados en los valores de los sliders
update_plot(k_dti, c_dti, dti_max, threshold)



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
