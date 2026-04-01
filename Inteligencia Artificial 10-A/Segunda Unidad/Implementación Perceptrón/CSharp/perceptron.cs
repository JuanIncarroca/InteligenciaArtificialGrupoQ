using System;
using System.Collections.Generic;
using System.Linq;
using ScottPlot; // Librería para graficar instalada vía NuGet

namespace PerceptronSpam
{
    class Program
    {
        // ==========================================
        // 1. FUNCIONES DE ACTIVACIÓN VIABLES
        // ==========================================
        public static double Step(double z) => z >= 0 ? 1.0 : 0.0;

        public static double Sigmoid(double z)
        {
            z = Math.Max(-700, Math.Min(700, z)); // Prevenir overflow
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        public static double[] SoftmaxSimulada(double z)
        {
            z = Math.Max(-700, Math.Min(700, z));
            double expNeg = Math.Exp(-z);
            double expPos = Math.Exp(z);
            double sum = expNeg + expPos;
            return new double[] { expNeg / sum, expPos / sum }; // [P(Normal), P(Spam)]
        }

        // ==========================================
        // 2. CLASE PERCEPTRÓN
        // ==========================================
        class Perceptron
        {
            public double[] weights;
            public double bias;
            public double learningRate = 0.1;
            public string activationName;

            public Perceptron(int inputs, string activation)
            {
                weights = new double[inputs]; // Inicializan en 0
                bias = 0.0;
                activationName = activation;
            }

            public double Activate(double z)
            {
                if (activationName == "step") return Step(z);
                else if (activationName == "sigmoid") return Sigmoid(z);
                else if (activationName == "softmax")
                {
                    var probs = SoftmaxSimulada(z);
                    return probs[1] > probs[0] ? 1.0 : 0.0;
                }
                return Step(z);
            }

            public double Predict(double[] inputs, out double rawOutput)
            {
                double z = bias;
                for (int i = 0; i < weights.Length; i++)
                    z += weights[i] * inputs[i];

                rawOutput = z; // Guardamos Z para calcular la "salida bruta" en la impresión
                return Activate(z);
            }

            public List<double> Train(double[][] X, double[] y, int epochs)
            {
                List<double> errorHistory = new List<double>();

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    double totalError = 0;
                    for (int i = 0; i < X.Length; i++)
                    {
                        double prediction = Predict(X[i], out _);

                        // Adaptación para calcular error
                        double predBin = prediction;
                        if (activationName == "sigmoid") predBin = prediction >= 0.5 ? 1.0 : 0.0;

                        double error = y[i] - predBin;
                        totalError += Math.Abs(error);

                        for (int w = 0; w < weights.Length; w++)
                            weights[w] += learningRate * error * X[i][w];
                        bias += learningRate * error;
                    }

                    errorHistory.Add(totalError);

                    if (totalError == 0 && (activationName == "step" || activationName == "softmax"))
                    {
                        Console.WriteLine($"Convergencia alcanzada en la época {epoch + 1}.");
                        while (errorHistory.Count < epochs) errorHistory.Add(0); // Rellenar para gráfica
                        break;
                    }
                    if (epoch == epochs - 1 && totalError == 0)
                        Console.WriteLine($"Convergencia alcanzada en la época {epoch + 1}.");
                }
                return errorHistory;
            }
        }

        // ==========================================
        // 3. EJECUCIÓN PRINCIPAL Y GRÁFICOS
        // ==========================================
        static void Main(string[] args)
        {
            // Datos: X = [Contiene "gratis", >3 signos "!"]
            double[][] X_Spam = {
                new double[]{0,0},
                new double[]{1,0},
                new double[]{0,1},
                new double[]{1,1}
            };
            double[] y_Spam = { 0, 1, 0, 1 };

            string[] funcionesViables = { "step", "sigmoid", "softmax" };
            Dictionary<string, List<double>> resultadosErrores = new Dictionary<string, List<double>>();

            Console.WriteLine("=== EVALUACIÓN DE DETECCIÓN DE SPAM ===\n");

            foreach (string func in funcionesViables)
            {
                Console.WriteLine($"--- Entrenando con activación: {func.ToUpper()} ---");
                Perceptron p = new Perceptron(2, func);
                List<double> errores = p.Train(X_Spam, y_Spam, 15);
                resultadosErrores[func] = errores;

                Console.WriteLine($"Pesos Finales: W=[{p.weights[0]:F1}, {p.weights[1]:F1}], Bias={p.bias:F1}");
                Console.WriteLine("Resultados:");

                for (int i = 0; i < X_Spam.Length; i++)
                {
                    double z;
                    p.Predict(X_Spam[i], out z); // Obtenemos la suma ponderada Z

                    if (func == "step")
                        Console.WriteLine($"Entrada [{X_Spam[i][0]}, {X_Spam[i][1]}] | Esperado: {y_Spam[i]} | Salida bruta: {Step(z)}");
                    else if (func == "sigmoid")
                        Console.WriteLine($"Entrada [{X_Spam[i][0]}, {X_Spam[i][1]}] | Esperado: {y_Spam[i]} | Salida bruta: {Sigmoid(z):F4}");
                    else if (func == "softmax")
                    {
                        var probs = SoftmaxSimulada(z);
                        double salidaBruta = probs[1] > probs[0] ? 1 : 0;
                        Console.WriteLine($"Entrada [{X_Spam[i][0]}, {X_Spam[i][1]}] | Esperado: {y_Spam[i]} | Salida bruta: {salidaBruta}");
                    }
                }
                Console.WriteLine();
            }

            // --- GENERAR GRÁFICOS CON SCOTTPLOT ---
            GenerarGraficoFunciones();
            GenerarGraficoError(resultadosErrores);

            Console.WriteLine("Las gráficas 'FuncionesActivacion.png' y 'CurvaAprendizaje.png' han sido guardadas en la carpeta de compilación (bin/Debug).");
            Console.WriteLine("\nPresiona cualquier tecla para salir...");
            Console.ReadKey();
        }

        // Métodos auxiliares para dibujar las gráficas
        static void GenerarGraficoFunciones()
        {
            var plt = new ScottPlot.Plot(600, 400);
            double[] z_vals = DataGen.Range(-10, 10, 0.2);
            double[] step_vals = z_vals.Select(z => Step(z)).ToArray();
            double[] sig_vals = z_vals.Select(z => Sigmoid(z)).ToArray();
            double[] soft_vals = z_vals.Select(z => SoftmaxSimulada(z)[1]).ToArray();

            plt.AddScatter(z_vals, step_vals, label: "Escalón (Step)");
            plt.AddScatter(z_vals, sig_vals, label: "Sigmoidal");
            plt.AddScatter(z_vals, soft_vals, label: "Softmax (Prob. Spam)", lineStyle: LineStyle.Dot);

            plt.Title("Comportamiento Matemático de las Funciones");
            plt.XLabel("Suma Ponderada (z)");
            plt.YLabel("Salida de Activación f(z)");
            plt.Legend();
            plt.SaveFig("FuncionesActivacion.png");
        }

        static void GenerarGraficoError(Dictionary<string, List<double>> resultadosErrores)
        {
            var plt = new ScottPlot.Plot(600, 400);
            double[] epocas = DataGen.Consecutive(15, 1); // 1 a 15

            plt.AddScatter(epocas, resultadosErrores["step"].ToArray(), label: "Escalón");
            plt.AddScatter(epocas, resultadosErrores["sigmoid"].ToArray(), label: "Sigmoidal");
            plt.AddScatter(epocas, resultadosErrores["softmax"].ToArray(), label: "Softmax");

            plt.Title("Curva de Aprendizaje: Error por Época");
            plt.XLabel("Número de Épocas");
            plt.YLabel("Error Total");
            plt.Legend();
            plt.SaveFig("CurvaAprendizaje.png");
        }
    }
}
