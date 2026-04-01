import math
import random

# ==========================================
# 3. FUNCIONES DE ACTIVACIÓN (Obligatorias)
# ==========================================
def step(z):
    return 1 if z >= 0 else 0

def lineal(z):
    return z

def sigmoid(z):
    # Prevenir overflow
    z = max(-700, min(700, z))
    return 1.0 / (1.0 + math.exp(-z))

def relu(z):
    return max(0, z)

def tanh_act(z):
    z = max(-700, min(700, z))
    return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))

def softmax_simulada(z):
    # Simulacion para dos clases [clase0, clase1]
    z = max(-700, min(700, z))
    exp_neg = math.exp(-z)
    exp_pos = math.exp(z)
    suma = exp_neg + exp_pos
    return [exp_neg / suma, exp_pos / suma]

class Perceptron:
    def __init__(self, num_inputs, activation_fn=step, is_softmax=False):
        # 2. INICIALIZACIÓN DE PESOS
        self.weights = [0.0 for _ in range(num_inputs)]
        self.bias = 0.0
        self.learning_rate = 0.1
        self.activation_fn = activation_fn
        self.is_softmax = is_softmax

    def predict(self, inputs):
        # 5. PREDICCIÓN (Suma ponderada + Activación)
        z = self.bias
        for i in range(len(inputs)):
            z += self.weights[i] * inputs[i]
        
        if self.is_softmax:
            probs = self.activation_fn(z)
            return 1 if probs[1] > probs[0] else 0
        
        return self.activation_fn(z)

    def train(self, X, y, epochs=15):
        # 4. ENTRENAMIENTO
        print(f"--- Pesos Iniciales: W={self.weights}, Bias={self.bias} ---")
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                # Predicción actual
                prediction = self.predict(inputs)
                
                # Para funciones continuas, convertimos a 0 o 1 usando un umbral para calcular el error
                # en este enfoque básico de perceptrón (Regla de Hebbian/Delta simple)
                if not self.is_softmax and self.activation_fn in [sigmoid, tanh_act, lineal]:
                    pred_binaria = 1 if prediction >= 0.5 else 0
                    error = target - pred_binaria
                else:
                    error = target - prediction
                
                total_error += abs(error)
                
                # Actualización de pesos
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error
                
            print(f"Época {epoch+1} - Error total: {total_error}")
            if total_error == 0:
                print("Convergencia alcanzada.")
                break
        print(f"--- Pesos Finales: W={self.weights}, Bias={self.bias} ---")

# 1. DECLARACIÓN DE DATOS (Casos de prueba)
casos = {
    "1. AND Lógico": {"X": [[0,0], [0,1], [1,0], [1,1]], "y": [0,0,0,1]},
    "2. OR Lógico": {"X": [[0,0], [0,1], [1,0], [1,1]], "y": [0,1,1,1]},
    "3. Spam (Gratis, >3 !!)": {"X": [[0,0], [1,0], [0,1], [1,1]], "y": [0,1,0,1]}, # Depende solo de la palabra "gratis"
    "4. Clima (Nublado, Humedad)": {"X": [[0,0], [0,1], [1,0], [1,1]], "y": [0,0,1,1]} # Depende solo de "Nublado"
}

# 6. RESULTADOS
print("=== EVALUACIÓN DE FUNCIONES DE ACTIVACIÓN (Con el caso AND) ===")
funciones = [
    ("Escalón", step, False), 
    ("Lineal (No apta para clasif binaria)", lineal, False),
    ("Sigmoide", sigmoid, False),
    ("ReLU", relu, False),
    ("Tanh", tanh_act, False),
    ("Softmax", softmax_simulada, True)
]

X_and = casos["1. AND Lógico"]["X"]
y_and = casos["1. AND Lógico"]["y"]

for nombre, func, is_smax in funciones:
    print(f"\nProbando función: {nombre}")
    p = Perceptron(2, activation_fn=func, is_softmax=is_smax)
    p.train(X_and, y_and, epochs=15)
    
print("\n=== RESOLUCIÓN DE LOS 4 CASOS (Usando función Escalón) ===")
for nombre, data in casos.items():
    print(f"\n--- Caso: {nombre} ---")
    p = Perceptron(2, activation_fn=step)
    p.train(data["X"], data["y"], epochs=15)
    print("Predicciones Finales:")
    for x, target in zip(data["X"], data["y"]):
        pred = p.predict(x)
        print(f"Entrada: {x} | Esperado: {target} | Predicción: {pred}")
