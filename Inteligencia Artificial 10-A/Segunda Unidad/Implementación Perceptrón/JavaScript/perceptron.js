// ==========================================
// CASO: DETECCIÓN DE FRAUDES (Lógica AND)
// ==========================================

// 3. FUNCIÓN DE ACTIVACIÓN: ESCALÓN
const stepFunction = (z) => (z >= 0 ? 1 : 0);

class Perceptron {
    constructor(numInputs) {
        // 2. INICIALIZACIÓN DE PESOS Y BIAS
        this.weights = new Array(numInputs).fill(0);
        this.bias = 0;
        this.learningRate = 0.1;
    }

    // 5. PREDICCIÓN (Suma ponderada + Activación)
    predict(inputs) {
        let z = this.bias;
        for (let i = 0; i < inputs.length; i++) {
            z += this.weights[i] * inputs[i];
        }
        return stepFunction(z);
    }

    // 4. ENTRENAMIENTO
    train(X, y, epochs = 15) {
        console.log(`--- Pesos Iniciales: W=[${this.weights}], Bias=${this.bias} ---`);
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            
            for (let i = 0; i < X.length; i++) {
                let prediction = this.predict(X[i]);
                let error = y[i] - prediction;
                totalError += Math.abs(error);

                // Actualización de pesos y bias
                for (let j = 0; j < this.weights.length; j++) {
                    this.weights[j] += this.learningRate * error * X[i][j];
                }
                this.bias += this.learningRate * error;
            }
            
            console.log(`Época ${epoch + 1} - Error total: ${totalError}`);
            if (totalError === 0) {
                console.log(">> Convergencia alcanzada.");
                break;
            }
        }
        console.log(`--- Pesos Finales: W=[${this.weights.map(w => w.toFixed(2))}], Bias=${this.bias.toFixed(2)} ---`);
    }
}

// 1. DECLARACIÓN DE DATOS (Monto inusual, Ubicación extraña)
const X_Fraude = [
    [0, 0], [0, 1], [1, 0], [1, 1]
];
const y_Fraude = [0, 0, 0, 1]; // Solo es fraude si ambas son 1 (AND)

// 6. RESULTADOS
const p = new Perceptron(2);
p.train(X_Fraude, y_Fraude, 20);

console.log("\n--- TABLA DE PREDICCIONES FINALES ---");
X_Fraude.forEach((input, index) => {
    const pred = p.predict(input);
    console.log(`Entrada: [${input}] | Esperado: ${y_Fraude[index]} | Predicho: ${pred}`);
});
