// Función de Activación Escalón
const stepFunction = (z) => z >= 0 ? 1 : 0;

class Perceptron {
    constructor(numInputs) {
        this.weights = new Array(numInputs).fill(0);
        this.bias = 0;
        this.learningRate = 0.1;
    }

    predict(inputs) {
        let z = this.bias;
        for (let i = 0; i < inputs.length; i++) {
            z += this.weights[i] * inputs[i];
        }
        return stepFunction(z);
    }

    train(X, y, epochs = 15) {
        console.log(`--- Pesos Iniciales: W=[${this.weights}], Bias=${this.bias} ---`);
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            
            for (let i = 0; i < X.length; i++) {
                let prediction = this.predict(X[i]);
                let error = y[i] - prediction;
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

// Datos: X = [Monto inusual, Ubicación extraña]
const X_Fraude = [
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1]
];
// Fraude solo si ambas variables son 1 (AND lógico)
const y_Fraude = [0, 0, 0, 1];

console.log("=== DETECCIÓN DE FRAUDES ===");
let p = new Perceptron(2);
p.train(X_Fraude, y_Fraude, 15);

console.log("\nPredicciones Finales:");
for (let i = 0; i < X_Fraude.length; i++) {
    let pred = p.predict(X_Fraude[i]);
    console.log(`Entrada: [${X_Fraude[i]}] | Esperado: ${y_Fraude[i]} | Predicción: ${pred}`);
}
