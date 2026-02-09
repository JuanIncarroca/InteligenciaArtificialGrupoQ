import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Datos de entrenamiento (convertidos a Tensores de punto flotante 32)
x_train = tf.constant([
    [50.0, 1.50], [60.0, 1.60], [70.0, 1.70], [80.0, 1.80], [90.0, 1.90],
    [45.0, 1.55], [65.0, 1.65], [75.0, 1.75], [85.0, 1.85], [100.0, 1.70]
], dtype=tf.float32)

y_train = tf.constant([
    [22.22], [23.44], [24.22], [24.69], [24.93],
    [18.73], [23.88], [24.49], [24.84], [34.60]
], dtype=tf.float32)

# 2. Definición de Variables (Pesos y Sesgo) - TensorFlow Neto
# Inicializamos pesos aleatorios para las 2 entradas y un sesgo
W = tf.Variable(tf.random.normal([2, 1]), name='pesos')
b = tf.Variable(tf.zeros([1]), name='sesgo')

# 3. Función del Modelo (Lineal: y = X*W + b)
def modelo_lineal(X):
    return tf.matmul(X, W) + b

# 4. Función de Pérdida (Error Cuadrático Medio)
def calcular_perdida(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# 5. Optimizador
optimizador = tf.optimizers.Adam(learning_rate=0.1)

print("--- FASE DE ENTRENAMIENTO (TF NETO) ---")

# Ciclo de entrenamiento manual
for epoch in range(2000):
    with tf.GradientTape() as tape:
        predicciones = modelo_lineal(x_train)
        perdida = calcular_perdida(predicciones, y_train)
    
    # Calculamos los gradientes respecto a W y b
    gradientes = tape.gradient(perdida, [W, b])
    
    # Aplicamos los gradientes para actualizar las variables
    optimizador.apply_gradients(zip(gradientes, [W, b]))
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Pérdida = {perdida.numpy():.4f}")

print("¡Entrenamiento completado!\n")

# --- BUCLE DE CONSULTAS ---
print("="*40)
print(" PREDICTOR DE IMC - TENSORFLOW PURO")
print(" (Escribe 'salir' para finalizar)")
print("="*40)

while True:
    entrada_usuario = input("\nPeso en kg (o 'salir'): ").lower()
    if entrada_usuario == 'salir': break
    
    try:
        peso = float(entrada_usuario)
        altura = float(input("Altura en metros: "))

        # Preparar el dato para la red
        x_input = tf.constant([[peso, altura]], dtype=tf.float32)
        
        # Realizar la predicción manual
        prediccion_tf = modelo_lineal(x_input)
        valor_imc = prediccion_tf.numpy()[0][0]
        
        # Comparar con fórmula real
        real = peso / (altura ** 2)

        print(f"{'-'*30}")
        print(f"IA (TF Neto): {valor_imc:.2f}")
        print(f"Fórmula Real: {real:.2f}")
        print(f"Diferencia:   {abs(valor_imc - real):.4f}")
        print(f"{'-'*30}")

    except Exception as e:
        print(f"Error: {e}")

print("Programa finalizado.")
