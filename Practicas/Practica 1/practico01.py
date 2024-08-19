import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generar datos aleatorios para la altura (m) y el peso (kg)
np.random.seed(0)
alturas = np.random.uniform(1.4, 2.0, 100)  # Alturas entre 1.4m y 2.0m
pesos = []

# Generar pesos aleatorios controlados basados en la altura
for altura in alturas:
    peso = np.random.uniform(18.5 * altura ** 2, 25 * altura ** 2)  # IMC entre 18.5 y 25
    pesos.append(peso)

# Crear un DataFrame para almacenar los datos
datos = pd.DataFrame({
    'Altura (m)': alturas,
    'Peso (kg)': pesos
})

# Definir una función para el modelo (lineal en este caso)
def modelo_lineal(x, a, b):
    return a * x + b

# Ajustar la curva a los datos
parametros_optimos, matriz_covarianza = curve_fit(modelo_lineal, datos['Altura (m)'], datos['Peso (kg)'])

# Obtener los parámetros de la línea ajustada
a, b = parametros_optimos

# Generar los pesos predichos
pesos_predichos = modelo_lineal(datos['Altura (m)'], a, b)

# Graficar los datos y la línea ajustada
plt.scatter(datos['Altura (m)'], datos['Peso (kg)'], label='Datos')
plt.plot(datos['Altura (m)'], pesos_predichos, color='red', label=f'Línea ajustada: y = {a:.2f}x + {b:.2f}')
plt.xlabel('Altura (m)')
plt.ylabel('Peso (kg)')
plt.title('Relación entre Altura y Peso con Línea Ajustada')
plt.legend()
plt.show()

#Repositorio en github
#https://github.com/richardcruz98/Inteligencia-Artificial-I