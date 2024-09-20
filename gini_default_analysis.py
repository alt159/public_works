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

# Simular datos de DTI
np.random.seed(42)
n_samples = 10000

# Función para actualizar el gráfico, la matriz de confusión, y el Gini basado en los parámetros
def update_plot(k_dti, c_dti, dti_max):
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

    # Graficar curva de densidad de DTI para No Default y Default
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df[df['default'] == 0]['dti'], label='Sin Default', color='blue', fill=True, alpha=0.5)
    sns.kdeplot(df[df['default'] == 1]['dti'], label='Con Default', color='red', fill=True, alpha=0.5)

    plt.xlabel('Ratio Deuda/Ingreso (DTI)')
    plt.ylabel('Densidad')
    plt.legend(loc='upper right')
    plt.title(f'Curva de Densidad de DTI para Default y Sin Default\nGini: {gini_value:.4f}')
    st.pyplot(plt)

    # Graficar curva Gini
    plot_gini_curve(df)

# Crear sliders interactivos
k_dti = st.sidebar.slider('Pendiente (k_DTI)', 0.01, 20.0, 1.0, 0.1)
c_dti = st.sidebar.slider('Desplazamiento (c_DTI)', 0.0, 5.0, 1.5, 0.1)
dti_max = st.sidebar.slider('Máximo DTI', 1.0, 10.0, 3.0, 0.1)

# Actualizar gráficos basados en los valores de los sliders
update_plot(k_dti, c_dti, dti_max)
