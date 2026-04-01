import math

# --- Funciones de Activación Obligatorias ---
def step(z): 
    return 1 if z >= 0 else 0

def lineal(z): 
    return z

def sigmoid(z): 
    z = max(-700, min(700, z)) # Evitar overflow
    return 1.0 / (1.0 + math.exp(-z))

def relu(z): 
    return max(0, z)

def tanh_act(z): 
    z = max(-700, min(700, z))
    return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))

def softmax_simulada(z):
    z = max(-700, min(700, z))
    exp_neg, exp_pos = math.exp(-z), math.exp(z)
    suma = exp_neg + exp_pos
    return [exp_neg / suma, exp_pos / suma] # [P(Normal), P(Riesgo)]

class Perceptron:
    def __init__(self, num_inputs, activation_name="step"):
        self.weights = [0.0] * num_inputs
        self.bias = 0.0
        self.learning_rate = 0.1
        self.activation_name = activation_name

    def activate(self, z):
        if self.activation_name == "step": return step(z)
        elif self.activation_name == "lineal": return lineal(z)
        elif self.activation_name == "sigmoid": return sigmoid(z)
        elif self.activation_name == "relu": return relu(z)
        elif self.activation_name == "tanh": return tanh_act(z)
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
        print(f"--- Entrenando con activación: {self.activation_name} ---")
        print(f"Pesos Iniciales: W={self.weights}, Bias={self.bias}")
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                
                # Adaptación para calcular error en activaciones continuas
                pred_bin = 1 if prediction >= 0.5 else 0 if type(prediction) in [float, int] else prediction
                error = target - pred_bin
                
                total_error += abs(error)
                
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error
                
            print(f"Época {epoch+1} - Error total: {total_error}")
            if total_error == 0 and self.activation_name in ["step", "softmax"]:
                print("Convergencia alcanzada.")
                break
        print(f"Pesos Finales: W={self.weights}, Bias={self.bias}\n")

# --- Datos: Alumnos con Riesgo Académico ---
# X = [Faltas altas, Notas bajas]
X_riesgo = [[0,0], [0,1], [1,0], [1,1]]
y_riesgo = [0, 1, 1, 1]  # OR lógico: Riesgo si falla en cualquiera

# Ejecución
print("=== CLASIFICACIÓN DE RIESGO ACADÉMICO ===")
p = Perceptron(2, "step")
p.train(X_riesgo, y_riesgo, 15)

print("Predicciones Finales (Step):")
for x, target in zip(X_riesgo, y_riesgo):
    print(f"Entrada (Faltas, Notas): {x} | Esperado: {target} | Predicho: {p.predict(x)}")
