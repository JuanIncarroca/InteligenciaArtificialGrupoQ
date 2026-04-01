// 3. FUNCIONES DE ACTIVACIÓN
const step = z => z >= 0 ? 1 : 0;
const lineal = z => z;
const sigmoid = z => 1 / (1 + Math.exp(-z));
const relu = z => Math.max(0, z);
const tanhAct = z => Math.tanh(z);
const softmaxSim = z => {
    let expNeg = Math.exp(-z);
    let expPos = Math.exp(z);
    let sum = expNeg + expPos;
    return [expNeg / sum, expPos / sum]; // [P(clase0), P(clase1)]
};

class Perceptron {
    constructor(numInputs, activationFn = step) {
        // 2. INICIALIZACIÓN DE PESOS
        this.weights = new Array(numInputs).fill(0);
        this.bias = 0;
        this.learningRate = 0.1;
        this.activationFn = activationFn;
    }

    predict(inputs) {
        // 5. PREDICCIÓN
        let z = this.bias;
        for (let i = 0; i < inputs.length; i++) {
            z += this.weights[i] * inputs[i];
        }
        return this.activationFn(z);
    }

    train(X, y, epochs = 15) {
        // 4. ENTRENAMIENTO
        console.log(`--- Pesos Iniciales: W=[${this.weights}], Bias=${this.bias} ---`);
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            for (let i = 0; i < X.length; i++) {
                let prediction = this.predict(X[i]);
                
                // Tratar salida de activaciones continuas como binaria para calcular el error
                let binPred = prediction >= 0.5 ? 1 : 0;
                let error = y[i] - binPred;
                
                totalError += Math.abs(error);

                for (let j = 0; j < this.weights.length; j++) {
                    this.weights[j] += this.learningRate * error * X[i][j];
                }
                this.bias += this.learningRate * error;
            }
            console.log(`Época ${epoch + 1} - Error total: ${totalError}`);
            if (totalError === 0) {
                console.log("Convergencia alcanzada.");
                break;
            }
        }
        console.log(`--- Pesos Finales: W=[${this.weights}], Bias=${this.bias} ---`);
    }
}

// 1. DECLARACIÓN DE DATOS
const X_OR = [[0,0], [0,1], [1,0], [1,1]];
const y_OR = [0, 1, 1, 1];

// 6. RESULTADOS
console.log("=== CASO 2: OR LÓGICO (Usando Escalón) ===");
let p = new Perceptron(2, step);
p.train(X_OR, y_OR, 15);

console.log("Predicciones Finales:");
for (let i = 0; i < X_OR.length; i++) {
    let pred = p.predict(X_OR[i]);
    console.log(`Entrada: [${X_OR[i]}] | Esperado: ${y_OR[i]} | Predicción: ${pred}`);
}
console.log("Explicación: El caso OR se resuelve ajustando el Bias con un valor negativo pequeño (-0.1) y pesos suficientemente grandes (0.1) para que si AL MENOS UNA entrada es 1, se supere el umbral de 0.");
