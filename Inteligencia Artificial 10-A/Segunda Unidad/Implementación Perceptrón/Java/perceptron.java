package climaperceptron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries.XYSeriesRenderStyle;
import org.knowm.xchart.style.markers.SeriesMarkers;

public class ClimaPerceptron {

    // ==========================================
    // 1. FUNCIONES DE ACTIVACIÓN VIABLES
    // ==========================================
    public static double stepFunction(double z) {
        return z >= 0 ? 1.0 : 0.0;
    }

    public static double sigmoid(double z) {
        z = Math.max(-700, Math.min(700, z)); // Prevenir overflow
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double[] softmaxSimulada(double z) {
        z = Math.max(-700, Math.min(700, z));
        double expNeg = Math.exp(-z);
        double expPos = Math.exp(z);
        double sum = expNeg + expPos;
        return new double[]{expNeg / sum, expPos / sum}; // [P(No Lloverá), P(Lloverá)]
    }

    // ==========================================
    // 2. CLASE PERCEPTRÓN
    // ==========================================
    static class Perceptron {
        public double[] weights;
        public double bias;
        public double learningRate = 0.1;
        public String activationName;

        public Perceptron(int numInputs, String activationName) {
            this.weights = new double[numInputs];
            this.bias = 0.0;
            this.activationName = activationName;
        }

        public double activate(double z) {
            switch (activationName) {
                case "step":
                    return stepFunction(z);
                case "sigmoid":
                    return sigmoid(z);
                case "softmax":
                    double[] probs = softmaxSimulada(z);
                    return probs[1] > probs[0] ? 1.0 : 0.0;
                default:
                    return stepFunction(z);
            }
        }

        // Retorna la salida activada y guarda Z en un array de 1 elemento (paso por referencia simulado)
        public double predict(double[] inputs, double[] rawZ) {
            double z = bias;
            for (int i = 0; i < weights.length; i++) {
                z += weights[i] * inputs[i];
            }
            rawZ[0] = z;
            return activate(z);
        }

        public List<Double> train(double[][] X, double[] y, int epochs) {
            List<Double> errorHistory = new ArrayList<>();

            for (int epoch = 0; epoch < epochs; epoch++) {
                double totalError = 0;
                for (int i = 0; i < X.length; i++) {
                    double[] rawZ = new double[1];
                    double prediction = predict(X[i], rawZ);

                    // Adaptación para calcular error
                    double predBin = prediction;
                    if (activationName.equals("sigmoid")) {
                        predBin = prediction >= 0.5 ? 1.0 : 0.0;
                    }

                    double error = y[i] - predBin;
                    totalError += Math.abs(error);

                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += learningRate * error * X[i][j];
                    }
                    bias += learningRate * error;
                }

                errorHistory.add(totalError);

                if (totalError == 0.0 && (activationName.equals("step") || activationName.equals("softmax"))) {
                    System.out.println("Convergencia alcanzada en la época " + (epoch + 1) + ".");
                    while (errorHistory.size() < epochs) {
                        errorHistory.add(0.0); // Rellenar para gráfica
                    }
                    break;
                }
                if (epoch == epochs - 1 && totalError == 0.0) {
                    System.out.println("Convergencia alcanzada en la época " + (epoch + 1) + ".");
                }
            }
            return errorHistory;
        }
    }

    // ==========================================
    // 3. EJECUCIÓN PRINCIPAL Y GRÁFICOS
    // ==========================================
    public static void main(String[] args) throws Exception {
        double[][] X_Clima = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] y_Clima = {0, 0, 1, 1};

        String[] funcionesViables = {"step", "sigmoid", "softmax"};
        Map<String, List<Double>> resultadosErrores = new HashMap<>();

        System.out.println("=== EVALUACIÓN DE PREDICCIÓN DEL CLIMA ===\n");

        for (String func : funcionesViables) {
            System.out.println("--- Entrenando con activación: " + func.toUpperCase() + " ---");
            Perceptron p = new Perceptron(2, func);
            List<Double> errores = p.train(X_Clima, y_Clima, 15);
            resultadosErrores.put(func, errores);

            System.out.printf("Pesos Finales: W=[%.1f, %.1f], Bias=%.1f\n", p.weights[0], p.weights[1], p.bias);
            System.out.println("Resultados:");

            for (int i = 0; i < X_Clima.length; i++) {
                double[] rawZ = new double[1];
                p.predict(X_Clima[i], rawZ);
                double z = rawZ[0];

                System.out.printf("Entrada [%.0f, %.0f] | Esperado: %.0f | Salida bruta: ", X_Clima[i][0], X_Clima[i][1], y_Clima[i]);
                
                if (func.equals("step")) {
                    System.out.printf("%.1f\n", stepFunction(z));
                } else if (func.equals("sigmoid")) {
                    System.out.printf("%.4f\n", sigmoid(z));
                } else if (func.equals("softmax")) {
                    double[] probs = softmaxSimulada(z);
                    System.out.printf("%.0f\n", probs[1] > probs[0] ? 1.0 : 0.0);
                }
            }
            System.out.println();
        }

        generarGraficoFunciones();
        generarGraficoError(resultadosErrores);
        
        System.out.println("Las gráficas 'FuncionesActivacion.png' y 'CurvaAprendizaje.png' han sido guardadas en la raíz del proyecto.");
    }

    // Métodos auxiliares para dibujar gráficas usando XChart
    public static void generarGraficoFunciones() throws Exception {
        XYChart chart = new XYChartBuilder().width(800).height(500).title("Comportamiento Matemático de las Funciones").xAxisTitle("Suma Ponderada (z)").yAxisTitle("Salida f(z)").build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Line);

        double[] zVals = new double[100];
        double[] stepVals = new double[100];
        double[] sigVals = new double[100];
        double[] softVals = new double[100];

        double z = -10.0;
        for (int i = 0; i < 100; i++) {
            zVals[i] = z;
            stepVals[i] = stepFunction(z);
            sigVals[i] = sigmoid(z);
            softVals[i] = softmaxSimulada(z)[1];
            z += 0.2;
        }

        chart.addSeries("Escalón (Step)", zVals, stepVals).setMarker(SeriesMarkers.NONE);
        chart.addSeries("Sigmoidal", zVals, sigVals).setMarker(SeriesMarkers.NONE);
        chart.addSeries("Softmax (Prob. Lloverá)", zVals, softVals).setMarker(SeriesMarkers.NONE);

        BitmapEncoder.saveBitmap(chart, "./FuncionesActivacion", BitmapFormat.PNG);
    }

    public static void generarGraficoError(Map<String, List<Double>> resultadosErrores) throws Exception {
        XYChart chart = new XYChartBuilder().width(800).height(500).title("Curva de Aprendizaje: Error por Época").xAxisTitle("Número de Épocas").yAxisTitle("Error Total").build();
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Line);

        double[] epocas = new double[15];
        for (int i = 0; i < 15; i++) {
            epocas[i] = i + 1;
        }

        for (Map.Entry<String, List<Double>> entry : resultadosErrores.entrySet()) {
            double[] errVals = new double[15];
            for (int i = 0; i < 15; i++) {
                errVals[i] = entry.getValue().get(i);
            }
            chart.addSeries(entry.getKey().toUpperCase(), epocas, errVals);
        }

        BitmapEncoder.saveBitmap(chart, "./CurvaAprendizaje", BitmapFormat.PNG);
    }
}
