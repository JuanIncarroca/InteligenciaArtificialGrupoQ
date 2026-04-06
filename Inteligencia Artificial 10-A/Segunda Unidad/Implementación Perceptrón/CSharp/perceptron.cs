using System;
namespace PerceptronSpam
{
    class Program
    {
        // Función de activación: Escalón (Ideal para binario)
        public static double Step(double z) => z >= 0 ? 1.0 : 0.0;
        class Perceptron
        {
            double[] weights;
            double bias;
            double learningRate = 0.1;
            public Perceptron(int inputs)
            {
                weights = new double[inputs]; // Inicializan en 0
                bias = 0.0;
            }
            public double Predict(double[] inputs)
            {
                double z = bias;
                for (int i = 0; i < weights.Length; i++)
                    z += weights[i] * inputs[i];
                return Step(z);
            }
            public void Train(double[][] X, double[] y, int epochs)
            {
                Console.WriteLine($"--- Pesos Iniciales: W=[{weights[0]}, {weights[1]}], Bias={bias} ---");
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    double totalError = 0;
                    for (int i = 0; i < X.Length; i++)
                    {
                        double pred = Predict(X[i]);
                        double error = y[i] - pred;
                        totalError += Math.Abs(error);
                        for (int w = 0; w < weights.Length; w++)
                            weights[w] += learningRate * error * X[i][w];
                        bias += learningRate * error;
                    }
                    Console.WriteLine($"Época {epoch + 1} - Error total: {totalError}");
                    if (totalError == 0)
                    {
                        Console.WriteLine("Convergencia alcanzada.");
                        break;
                    }
                }
                Console.WriteLine($"--- Pesos Finales: W=[{weights[0]}, {weights[1]}], Bias={bias} ---");
            }
        }
        static void Main(string[] args)
        {
            // Datos: X = [Contiene "gratis", >3 signos "!"]
            double[][] X_Spam = {
                new double[]{0,0},
                new double[]{1,0},
                new double[]{0,1},
                new double[]{1,1}
            };
            // El spam depende únicamente de la primera variable
            double[] y_Spam = { 0, 1, 0, 1 };
            Console.WriteLine("=== DETECCIÓN DE SPAM ===");
            Perceptron p = new Perceptron(2);
            p.Train(X_Spam, y_Spam, 15);
            Console.WriteLine("\nPredicciones Finales:");
            for (int i = 0; i < X_Spam.Length; i++)
            {
                double pred = p.Predict(X_Spam[i]);
                Console.WriteLine($"Entrada: [{X_Spam[i][0]}, {X_Spam[i][1]}] | Esperado: {y_Spam[i]} | Predicción: {pred}");
            }
            Console.WriteLine("\nPresiona cualquier tecla para salir...");
            Console.ReadKey();
        }
    }

}
