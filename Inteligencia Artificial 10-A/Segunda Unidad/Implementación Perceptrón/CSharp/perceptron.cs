using System;
using System.Linq;

namespace PerceptronSimple
{
    class Program
    {
        // 3. FUNCIÓN DE ACTIVACIÓN
        public static double Step(double z) => z >= 0 ? 1 : 0;
        public static double Linear(double z) => z;
        public static double Sigmoid(double z) => 1.0 / (1.0 + Math.Exp(-z));
        public static double Relu(double z) => Math.Max(0, z);
        public static double TanhAct(double z) => Math.Tanh(z);
        
        // Softmax simulada
        public static double[] Softmax(double z)
        {
            double expNeg = Math.Exp(-z);
            double expPos = Math.Exp(z);
            double sum = expNeg + expPos;
            return new double[] { expNeg / sum, expPos / sum };
        }

        class Perceptron
        {
            // 2. INICIALIZACIÓN DE PESOS
            double[] weights;
            double bias;
            double learningRate = 0.1;
            Func<double, double> activationFn;

            public Perceptron(int inputs, Func<double, double> activation)
            {
                weights = new double[inputs]; // Inicializan en 0
                bias = 0.0;
                activationFn = activation;
            }

            public double Predict(double[] inputs)
            {
                // 5. PREDICCIÓN
                double z = bias;
                for (int i = 0; i < weights.Length; i++)
                    z += weights[i] * inputs[i];
                
                return activationFn(z);
            }

            public void Train(double[][] X, double[] y, int epochs = 15)
            {
                // 4. ENTRENAMIENTO
                Console.WriteLine($"--- Pesos Iniciales: W=[{string.Join(", ", weights)}], Bias={bias} ---");
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    double totalError = 0;
                    for (int i = 0; i < X.Length; i++)
                    {
                        double pred = Predict(X[i]);
                        // Convertir a binario si la función es continua para calcular el error simple
                        double binaryPred = pred >= 0.5 ? 1 : 0; 
                        
                        double error = y[i] - binaryPred;
                        totalError += Math.Abs(error);

                        for (int w = 0; w < weights.Length; w++)
                            weights[w] += learningRate * error * X[i][w];
                        bias += learningRate * error;
                    }
                    Console.WriteLine($"Época {epoch + 1} - Error total: {totalError}");
                    if (totalError == 0) break;
                }
                Console.WriteLine($"--- Pesos Finales: W=[{string.Join(", ", weights)}], Bias={bias} ---");
            }
        }

        static void Main(string[] args)
        {
            // 1. DECLARACIÓN DE DATOS
            double[][] X_Spam = { new double[]{0,0}, new double[]{1,0}, new double[]{0,1}, new double[]{1,1} };
            double[] y_Spam = { 0, 1, 0, 1 };

            // 6. RESULTADOS
            Console.WriteLine("=== CASO 3: CLASIFICACIÓN SPAM (Usando Escalón) ===");
            Perceptron p = new Perceptron(2, Step);
            p.Train(X_Spam, y_Spam, 15);

            Console.WriteLine("Predicciones Finales:");
            for (int i = 0; i < X_Spam.Length; i++)
            {
                double pred = p.Predict(X_Spam[i]);
                Console.WriteLine($"Entrada: [{X_Spam[i][0]}, {X_Spam[i][1]}] | Esperado: {y_Spam[i]} | Predicción: {pred}");
            }
            Console.WriteLine("Explicación: Este problema es linealmente separable porque el perceptrón aprende a ignorar la característica 2 (signos de exclamación) poniendo su peso a 0, y le da peso positivo a la característica 1 (palabra gratis).");
        }
    }
}
