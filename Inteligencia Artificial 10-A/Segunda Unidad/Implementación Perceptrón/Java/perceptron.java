package climaperceptron;

public class ClimaPerceptron {

    // 3. FUNCIÓN DE ACTIVACIÓN: ESCALÓN
    public static double stepFunction(double z) {
        return z >= 0 ? 1.0 : 0.0;
    }

    // Clase interna del Perceptrón
    static class Perceptron {
        private double[] weights;
        private double bias;
        private double learningRate = 0.1;

        public Perceptron(int numInputs) {
            this.weights = new double[numInputs]; // Inicializa en 0
            this.bias = 0.0;
        }

        // 5. PREDICCIÓN (Suma ponderada + Activación)
        public double predict(double[] inputs) {
            double z = bias;
            for (int i = 0; i < weights.length; i++) {
                z += weights[i] * inputs[i];
            }
            return stepFunction(z);
        }

        // 4. ENTRENAMIENTO
        public void train(double[][] X, double[] y, int epochs) {
            System.out.println("--- Entrenamiento del Perceptrón (Clima) ---");
            for (int epoch = 0; epoch < epochs; epoch++) {
                double totalError = 0;
                for (int i = 0; i < X.length; i++) {
                    double prediction = predict(X[i]);
                    double error = y[i] - prediction;
                    totalError += Math.abs(error);

                    // Actualización de pesos y bias
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += learningRate * error * X[i][j];
                    }
                    bias += learningRate * error;
                }
                System.out.println("Época " + (epoch + 1) + " - Error total: " + totalError);
                if (totalError == 0.0) break;
            }
        }
    }

    public static void main(String[] args) {
        // 1. DATOS: X = [Nublado (0/1), Humedad Alta (0/1)]
        double[][] X_Clima = { {0,0}, {0,1}, {1,0}, {1,1} };
        // y = Lloverá (1) solo si está Nublado (Independiente de la humedad)
        double[] y_Clima = { 0, 0, 1, 1 };

        Perceptron p = new Perceptron(2);
        
        // Ejecutar entrenamiento
        p.train(X_Clima, y_Clima, 15);

        // 6. RESULTADOS FINALES
        System.out.println("\n--- RESULTADOS FINALES ---");
        for (int i = 0; i < X_Clima.length; i++) {
            double pred = p.predict(X_Clima[i]);
            System.out.println("Entrada: [" + (int)X_Clima[i][0] + "," + (int)X_Clima[i][1] + 
                               "] | Esperado: " + (int)y_Clima[i] + " | Predicho: " + (int)pred);
        }
    }
}
