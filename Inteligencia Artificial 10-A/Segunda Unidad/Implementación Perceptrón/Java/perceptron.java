// 3. FUNCIÓN DE ACTIVACIÓN
interface ActivationFunction {
    double apply(double z);
}

class Step implements ActivationFunction {
    public double apply(double z) { return z >= 0 ? 1.0 : 0.0; }
}

class Perceptron {
    // 2. INICIALIZACIÓN DE PESOS
    private double[] weights;
    private double bias;
    private double learningRate = 0.1;
    private ActivationFunction activation;

    public Perceptron(int numInputs, ActivationFunction activation) {
        this.weights = new double[numInputs];
        this.bias = 0.0;
        this.activation = activation;
    }

    // 5. PREDICCIÓN
    public double predict(double[] inputs) {
        double z = bias;
        for (int i = 0; i < weights.length; i++) {
            z += weights[i] * inputs[i];
        }
        return activation.apply(z);
    }

    // 4. ENTRENAMIENTO
    public void train(double[][] X, double[] y, int epochs) {
        System.out.printf("--- Pesos Iniciales: W=[%f, %f], Bias=%f ---\n", weights[0], weights[1], bias);
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
        System.out.printf("--- Pesos Finales: W=[%f, %f], Bias=%f ---\n", weights[0], weights[1], bias);
    }
}

public class Main {
    public static void main(String[] args) {
        // 1. DECLARACIÓN DE DATOS
        double[][] X_Clima = { {0,0}, {0,1}, {1,0}, {1,1} };
        double[] y_Clima = { 0, 0, 1, 1 };

        // 6. RESULTADOS
        System.out.println("=== CASO 4: PREDICCIÓN DE CLIMA (Usando Escalón) ===");
        Perceptron p = new Perceptron(2, new Step());
        p.train(X_Clima, y_Clima, 15);

        System.out.println("Predicciones Finales:");
        for (int i = 0; i < X_Clima.length; i++) {
            double pred = p.predict(X_Clima[i]);
            System.out.printf("Entrada: [%.0f, %.0f] | Esperado: %.0f | Predicción: %.0f\n", 
                              X_Clima[i][0], X_Clima[i][1], y_Clima[i], pred);
        }
        System.out.println("Explicación: El perceptrón se comporta aislando la variable 'Nublado' (x1) y asignándole un peso positivo, mientras que neutraliza el impacto de la humedad (x2).");
    }
}
