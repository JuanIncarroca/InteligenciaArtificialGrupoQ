import math
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNCIONES DE ACTIVACIÓN
# ==========================================
def step(z):
    return 1 if z >= 0 else 0

def sigmoid(z):
    z = max(-700, min(700, z)) # Evitar overflow
    return 1.0 / (1.0 + math.exp(-z))

def softmax_simulada(z):
    z = max(-700, min(700, z))
    exp_neg, exp_pos = math.exp(-z), math.exp(z)
    suma = exp_neg + exp_pos
    return [exp_neg / suma, exp_pos / suma] # [P(Normal), P(Riesgo)]

# ==========================================
# 2. CLASE PERCEPTRÓN
# ==========================================
class Perceptron:
    def __init__(self, num_inputs, activation_name="step"):
        self.weights = [0.0] * num_inputs
        self.bias = 0.0
        self.learning_rate = 0.1
        self.activation_name = activation_name

    def activate(self, z):
        if self.activation_name == "step": return step(z)
        elif self.activation_name == "sigmoid": return sigmoid(z)
        elif self.activation_name == "softmax":
            probs = softmax_simulada(z)
            return 1 if probs[1] > probs[0] else 0
        return step(z)

    def predict(self, inputs):
        z = self.bias
        for i in range(len(inputs)):
            z += self.weights[i] * inputs[i]
        return self.activate(z)

    def train(self, X, y, epochs=15):
        historial_errores = [] 
        
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)

                # Adaptación para calcular error
                pred_bin = 1 if prediction >= 0.5 else 0 if type(prediction) in [float, int] else prediction
                error = target - pred_bin

                total_error += abs(error)

                # Actualizar pesos
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error

            historial_errores.append(total_error)
            
            if total_error == 0:
                print(f"Convergencia alcanzada en la época {epoch+1}.")
                # Rellenar el resto del historial con 0 para la gráfica
                while len(historial_errores) < epochs:
                    historial_errores.append(0)
                break
                
        return historial_errores

# ==========================================
# 3. DATOS Y EJECUCIÓN (Salida Solicitada)
# ==========================================
X_riesgo = [[0,0], [0,1], [1,0], [1,1]]
y_riesgo = [0, 1, 1, 1]  

funciones_viables = ["step", "sigmoid", "softmax"]
resultados_errores = {}

print("=== EVALUACIÓN DE ALUMNOS CON RIESGO ACADÉMICO ===")

for func in funciones_viables:
    print(f"\n--- Entrenando con activación: {func.upper()} ---")
    p = Perceptron(2, func)
    errores = p.train(X_riesgo, y_riesgo, 15)
    resultados_errores[func] = errores
    
    # Formatear la salida de pesos y bias para que coincida con tu formato
    print(f"Pesos Finales: W=[{p.weights[0]:.1f}, {p.weights[1]:.1f}], Bias={p.bias:.1f}")
    print("Resultados:")
    
    for x, target in zip(X_riesgo, y_riesgo):
        # Calcular z manualmente para extraer la "salida bruta"
        z = p.bias + (p.weights[0] * x[0]) + (p.weights[1] * x[1])
        
        if func == "step":
            salida_bruta = step(z)
            print(f"Entrada {x} | Esperado: {target} | Salida bruta: {salida_bruta}")
        elif func == "sigmoid":
            salida_bruta = sigmoid(z)
            print(f"Entrada {x} | Esperado: {target} | Salida bruta: {salida_bruta:.4f}")
        elif func == "softmax":
            probs = softmax_simulada(z)
            salida_bruta = 1 if probs[1] > probs[0] else 0
            print(f"Entrada {x} | Esperado: {target} | Salida bruta: {salida_bruta}")

# ==========================================
# 4. GENERACIÓN DE GRÁFICOS (Matplotlib)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Formas de las Funciones de Activación
z_vals = np.linspace(-10, 10, 100)
step_vals = [step(z) for z in z_vals]
sig_vals = [sigmoid(z) for z in z_vals]
soft_vals = [softmax_simulada(z)[1] for z in z_vals]

ax1.plot(z_vals, step_vals, label="Escalón (Step)", color='blue', linewidth=2)
ax1.plot(z_vals, sig_vals, label="Sigmoidal", color='red', linestyle='--')
ax1.plot(z_vals, soft_vals, label="Softmax (Prob. Clase 1)", color='green', linestyle=':')
ax1.set_title("Comportamiento Matemático de las Funciones")
ax1.set_xlabel("Suma Ponderada (z)")
ax1.set_ylabel("Salida de Activación f(z)")
ax1.grid(True)
ax1.legend()

# Gráfico 2: Convergencia del Error durante el Entrenamiento
epocas = range(1, 16)
ax2.plot(epocas, resultados_errores["step"], marker='o', label="Escalón", color='blue')
ax2.plot(epocas, resultados_errores["sigmoid"], marker='s', label="Sigmoidal", color='red')
ax2.plot(epocas, resultados_errores["softmax"], marker='^', label="Softmax", color='green')

ax2.set_title("Curva de Aprendizaje: Error por Época")
ax2.set_xlabel("Número de Épocas")
ax2.set_ylabel("Error Total")
ax2.set_xticks(epocas)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
