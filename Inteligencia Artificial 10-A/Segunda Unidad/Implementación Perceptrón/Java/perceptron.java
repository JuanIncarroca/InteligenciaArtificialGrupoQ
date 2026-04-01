public class ClimaPerceptron {

    // Función de Activación Escalón
    public static double stepFunction(double z) {
        return z >= 0 ? 1.0 : 0.0;
    }

    static class Perceptron {
        private double[] weights;
        private double bias;
        private double learningRate = 0.1;

        public Perceptron(int numInputs) {
            this.weights = new double[numInputs];
            this.bias = 0.0;
        }

        public double predict(double[] inputs) {
            double z = bias;
            for (int i = 0; i < weights.length; i++) {
                z += weights[i] * inputs[i];
            }
            return stepFunction(z);
        }

        public void train(double[][] X, double[] y, int epochs) {
            System.out.printf("--- Pesos Iniciales: W=[%.2f, %.2f], Bias=%.2f ---\n", weights[0], weights[1], bias);
            
            for (int epoch = 0; epoch < epochs; epoch++) {
                double totalError = 0;
                
                for (int i = 0; i < X.length; i++) {
                    double prediction = predict(X[i]);
                    double error = y[i] - prediction;
                    totalError += Math.abs(error);

                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += learningRate * error * X[i][j];
                    }
                    bias += learningRate * error;
                }
                
                System.out.println("Época " + (epoch + 1) + " - Error total: " + totalError);
                if (totalError == 0.0) {
                    System.out.println("Convergencia alcanzada.");
                    break;
                }
            }
            System.out.printf("--- Pesos Finales: W=[%.2f, %.2f], Bias=%.2f ---\n", weights[0], weights[1], bias);
        }
    }

    public static void main(String[] args) {
        // Datos: X = [Nublado, Humedad Alta]
        double[][] X_Clima = { {0,0}, {0,1}, {1,0}, {1,1} };
        // Lloverá solo si está nublado (variable 1)
        double[] y_Clima = { 0, 0, 1, 1 };

        System.out.println("=== PREDICCIÓN DEL CLIMA ===");
        Perceptron p = new Perceptron(2);
        p.train(X_Clima, y_Clima, 15);

        System.out.println("\nPredicciones Finales:");
        for (int i = 0; i < X_Clima.length; i++) {
            double pred = p.predict(X_Clima[i]);
            System.out.printf("Entrada (Nublado, Humedad): [%.0f, %.0f] | Esperado: %.0f | Predicho: %.0f\n", 
                              X_Clima[i][0], X_Clima[i][1], y_Clima[i], pred);
        }
    }
}
