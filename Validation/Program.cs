using System;
using System.Configuration;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Common;
using NeuralNetwork;
using ParticleSwarm;

namespace Validation
{
	internal class Program
	{
		private static void Main ()
		{
			Console.WriteLine("\nBegin neural network demo\n");

			#region TrainData
			
			double[][] normalizedData = new double[150][];
			normalizedData[0] = new double[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 }; // sepal length, sepal width, petal length, petal width -> 
			normalizedData[1] = new double[] { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 }; // Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0
			normalizedData[2] = new double[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 };
			normalizedData[3] = new double[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 };
			normalizedData[4] = new double[] { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
			normalizedData[5] = new double[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
			normalizedData[6] = new double[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
			normalizedData[7] = new double[] { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
			normalizedData[8] = new double[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
			normalizedData[9] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };

			normalizedData[10] = new double[] { 5.4, 3.7, 1.5, 0.2, 0, 0, 1 };
			normalizedData[11] = new double[] { 4.8, 3.4, 1.6, 0.2, 0, 0, 1 };
			normalizedData[12] = new double[] { 4.8, 3.0, 1.4, 0.1, 0, 0, 1 };
			normalizedData[13] = new double[] { 4.3, 3.0, 1.1, 0.1, 0, 0, 1 };
			normalizedData[14] = new double[] { 5.8, 4.0, 1.2, 0.2, 0, 0, 1 };
			normalizedData[15] = new double[] { 5.7, 4.4, 1.5, 0.4, 0, 0, 1 };
			normalizedData[16] = new double[] { 5.4, 3.9, 1.3, 0.4, 0, 0, 1 };
			normalizedData[17] = new double[] { 5.1, 3.5, 1.4, 0.3, 0, 0, 1 };
			normalizedData[18] = new double[] { 5.7, 3.8, 1.7, 0.3, 0, 0, 1 };
			normalizedData[19] = new double[] { 5.1, 3.8, 1.5, 0.3, 0, 0, 1 };

			normalizedData[20] = new double[] { 5.4, 3.4, 1.7, 0.2, 0, 0, 1 };
			normalizedData[21] = new double[] { 5.1, 3.7, 1.5, 0.4, 0, 0, 1 };
			normalizedData[22] = new double[] { 4.6, 3.6, 1.0, 0.2, 0, 0, 1 };
			normalizedData[23] = new double[] { 5.1, 3.3, 1.7, 0.5, 0, 0, 1 };
			normalizedData[24] = new double[] { 4.8, 3.4, 1.9, 0.2, 0, 0, 1 };
			normalizedData[25] = new double[] { 5.0, 3.0, 1.6, 0.2, 0, 0, 1 };
			normalizedData[26] = new double[] { 5.0, 3.4, 1.6, 0.4, 0, 0, 1 };
			normalizedData[27] = new double[] { 5.2, 3.5, 1.5, 0.2, 0, 0, 1 };
			normalizedData[28] = new double[] { 5.2, 3.4, 1.4, 0.2, 0, 0, 1 };
			normalizedData[29] = new double[] { 4.7, 3.2, 1.6, 0.2, 0, 0, 1 };

			normalizedData[30] = new double[] { 4.8, 3.1, 1.6, 0.2, 0, 0, 1 };
			normalizedData[31] = new double[] { 5.4, 3.4, 1.5, 0.4, 0, 0, 1 };
			normalizedData[32] = new double[] { 5.2, 4.1, 1.5, 0.1, 0, 0, 1 };
			normalizedData[33] = new double[] { 5.5, 4.2, 1.4, 0.2, 0, 0, 1 };
			normalizedData[34] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
			normalizedData[35] = new double[] { 5.0, 3.2, 1.2, 0.2, 0, 0, 1 };
			normalizedData[36] = new double[] { 5.5, 3.5, 1.3, 0.2, 0, 0, 1 };
			normalizedData[37] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
			normalizedData[38] = new double[] { 4.4, 3.0, 1.3, 0.2, 0, 0, 1 };
			normalizedData[39] = new double[] { 5.1, 3.4, 1.5, 0.2, 0, 0, 1 };

			normalizedData[40] = new double[] { 5.0, 3.5, 1.3, 0.3, 0, 0, 1 };
			normalizedData[41] = new double[] { 4.5, 2.3, 1.3, 0.3, 0, 0, 1 };
			normalizedData[42] = new double[] { 4.4, 3.2, 1.3, 0.2, 0, 0, 1 };
			normalizedData[43] = new double[] { 5.0, 3.5, 1.6, 0.6, 0, 0, 1 };
			normalizedData[44] = new double[] { 5.1, 3.8, 1.9, 0.4, 0, 0, 1 };
			normalizedData[45] = new double[] { 4.8, 3.0, 1.4, 0.3, 0, 0, 1 };
			normalizedData[46] = new double[] { 5.1, 3.8, 1.6, 0.2, 0, 0, 1 };
			normalizedData[47] = new double[] { 4.6, 3.2, 1.4, 0.2, 0, 0, 1 };
			normalizedData[48] = new double[] { 5.3, 3.7, 1.5, 0.2, 0, 0, 1 };
			normalizedData[49] = new double[] { 5.0, 3.3, 1.4, 0.2, 0, 0, 1 };

			normalizedData[50] = new double[] { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
			normalizedData[51] = new double[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
			normalizedData[52] = new double[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
			normalizedData[53] = new double[] { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
			normalizedData[54] = new double[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
			normalizedData[55] = new double[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
			normalizedData[56] = new double[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
			normalizedData[57] = new double[] { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
			normalizedData[58] = new double[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
			normalizedData[59] = new double[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };

			normalizedData[60] = new double[] { 5.0, 2.0, 3.5, 1.0, 0, 1, 0 };
			normalizedData[61] = new double[] { 5.9, 3.0, 4.2, 1.5, 0, 1, 0 };
			normalizedData[62] = new double[] { 6.0, 2.2, 4.0, 1.0, 0, 1, 0 };
			normalizedData[63] = new double[] { 6.1, 2.9, 4.7, 1.4, 0, 1, 0 };
			normalizedData[64] = new double[] { 5.6, 2.9, 3.6, 1.3, 0, 1, 0 };
			normalizedData[65] = new double[] { 6.7, 3.1, 4.4, 1.4, 0, 1, 0 };
			normalizedData[66] = new double[] { 5.6, 3.0, 4.5, 1.5, 0, 1, 0 };
			normalizedData[67] = new double[] { 5.8, 2.7, 4.1, 1.0, 0, 1, 0 };
			normalizedData[68] = new double[] { 6.2, 2.2, 4.5, 1.5, 0, 1, 0 };
			normalizedData[69] = new double[] { 5.6, 2.5, 3.9, 1.1, 0, 1, 0 };

			normalizedData[70] = new double[] { 5.9, 3.2, 4.8, 1.8, 0, 1, 0 };
			normalizedData[71] = new double[] { 6.1, 2.8, 4.0, 1.3, 0, 1, 0 };
			normalizedData[72] = new double[] { 6.3, 2.5, 4.9, 1.5, 0, 1, 0 };
			normalizedData[73] = new double[] { 6.1, 2.8, 4.7, 1.2, 0, 1, 0 };
			normalizedData[74] = new double[] { 6.4, 2.9, 4.3, 1.3, 0, 1, 0 };
			normalizedData[75] = new double[] { 6.6, 3.0, 4.4, 1.4, 0, 1, 0 };
			normalizedData[76] = new double[] { 6.8, 2.8, 4.8, 1.4, 0, 1, 0 };
			normalizedData[77] = new double[] { 6.7, 3.0, 5.0, 1.7, 0, 1, 0 };
			normalizedData[78] = new double[] { 6.0, 2.9, 4.5, 1.5, 0, 1, 0 };
			normalizedData[79] = new double[] { 5.7, 2.6, 3.5, 1.0, 0, 1, 0 };

			normalizedData[80] = new double[] { 5.5, 2.4, 3.8, 1.1, 0, 1, 0 };
			normalizedData[81] = new double[] { 5.5, 2.4, 3.7, 1.0, 0, 1, 0 };
			normalizedData[82] = new double[] { 5.8, 2.7, 3.9, 1.2, 0, 1, 0 };
			normalizedData[83] = new double[] { 6.0, 2.7, 5.1, 1.6, 0, 1, 0 };
			normalizedData[84] = new double[] { 5.4, 3.0, 4.5, 1.5, 0, 1, 0 };
			normalizedData[85] = new double[] { 6.0, 3.4, 4.5, 1.6, 0, 1, 0 };
			normalizedData[86] = new double[] { 6.7, 3.1, 4.7, 1.5, 0, 1, 0 };
			normalizedData[87] = new double[] { 6.3, 2.3, 4.4, 1.3, 0, 1, 0 };
			normalizedData[88] = new double[] { 5.6, 3.0, 4.1, 1.3, 0, 1, 0 };
			normalizedData[89] = new double[] { 5.5, 2.5, 4.0, 1.3, 0, 1, 0 };

			normalizedData[90] = new double[] { 5.5, 2.6, 4.4, 1.2, 0, 1, 0 };
			normalizedData[91] = new double[] { 6.1, 3.0, 4.6, 1.4, 0, 1, 0 };
			normalizedData[92] = new double[] { 5.8, 2.6, 4.0, 1.2, 0, 1, 0 };
			normalizedData[93] = new double[] { 5.0, 2.3, 3.3, 1.0, 0, 1, 0 };
			normalizedData[94] = new double[] { 5.6, 2.7, 4.2, 1.3, 0, 1, 0 };
			normalizedData[95] = new double[] { 5.7, 3.0, 4.2, 1.2, 0, 1, 0 };
			normalizedData[96] = new double[] { 5.7, 2.9, 4.2, 1.3, 0, 1, 0 };
			normalizedData[97] = new double[] { 6.2, 2.9, 4.3, 1.3, 0, 1, 0 };
			normalizedData[98] = new double[] { 5.1, 2.5, 3.0, 1.1, 0, 1, 0 };
			normalizedData[99] = new double[] { 5.7, 2.8, 4.1, 1.3, 0, 1, 0 };

			normalizedData[100] = new double[] { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
			normalizedData[101] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
			normalizedData[102] = new double[] { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
			normalizedData[103] = new double[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
			normalizedData[104] = new double[] { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
			normalizedData[105] = new double[] { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
			normalizedData[106] = new double[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
			normalizedData[107] = new double[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
			normalizedData[108] = new double[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
			normalizedData[109] = new double[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };

			normalizedData[110] = new double[] { 6.5, 3.2, 5.1, 2.0, 1, 0, 0 };
			normalizedData[111] = new double[] { 6.4, 2.7, 5.3, 1.9, 1, 0, 0 };
			normalizedData[112] = new double[] { 6.8, 3.0, 5.5, 2.1, 1, 0, 0 };
			normalizedData[113] = new double[] { 5.7, 2.5, 5.0, 2.0, 1, 0, 0 };
			normalizedData[114] = new double[] { 5.8, 2.8, 5.1, 2.4, 1, 0, 0 };
			normalizedData[115] = new double[] { 6.4, 3.2, 5.3, 2.3, 1, 0, 0 };
			normalizedData[116] = new double[] { 6.5, 3.0, 5.5, 1.8, 1, 0, 0 };
			normalizedData[117] = new double[] { 7.7, 3.8, 6.7, 2.2, 1, 0, 0 };
			normalizedData[118] = new double[] { 7.7, 2.6, 6.9, 2.3, 1, 0, 0 };
			normalizedData[119] = new double[] { 6.0, 2.2, 5.0, 1.5, 1, 0, 0 };

			normalizedData[120] = new double[] { 6.9, 3.2, 5.7, 2.3, 1, 0, 0 };
			normalizedData[121] = new double[] { 5.6, 2.8, 4.9, 2.0, 1, 0, 0 };
			normalizedData[122] = new double[] { 7.7, 2.8, 6.7, 2.0, 1, 0, 0 };
			normalizedData[123] = new double[] { 6.3, 2.7, 4.9, 1.8, 1, 0, 0 };
			normalizedData[124] = new double[] { 6.7, 3.3, 5.7, 2.1, 1, 0, 0 };
			normalizedData[125] = new double[] { 7.2, 3.2, 6.0, 1.8, 1, 0, 0 };
			normalizedData[126] = new double[] { 6.2, 2.8, 4.8, 1.8, 1, 0, 0 };
			normalizedData[127] = new double[] { 6.1, 3.0, 4.9, 1.8, 1, 0, 0 };
			normalizedData[128] = new double[] { 6.4, 2.8, 5.6, 2.1, 1, 0, 0 };
			normalizedData[129] = new double[] { 7.2, 3.0, 5.8, 1.6, 1, 0, 0 };

			normalizedData[130] = new double[] { 7.4, 2.8, 6.1, 1.9, 1, 0, 0 };
			normalizedData[131] = new double[] { 7.9, 3.8, 6.4, 2.0, 1, 0, 0 };
			normalizedData[132] = new double[] { 6.4, 2.8, 5.6, 2.2, 1, 0, 0 };
			normalizedData[133] = new double[] { 6.3, 2.8, 5.1, 1.5, 1, 0, 0 };
			normalizedData[134] = new double[] { 6.1, 2.6, 5.6, 1.4, 1, 0, 0 };
			normalizedData[135] = new double[] { 7.7, 3.0, 6.1, 2.3, 1, 0, 0 };
			normalizedData[136] = new double[] { 6.3, 3.4, 5.6, 2.4, 1, 0, 0 };
			normalizedData[137] = new double[] { 6.4, 3.1, 5.5, 1.8, 1, 0, 0 };
			normalizedData[138] = new double[] { 6.0, 3.0, 4.8, 1.8, 1, 0, 0 };
			normalizedData[139] = new double[] { 6.9, 3.1, 5.4, 2.1, 1, 0, 0 };

			normalizedData[140] = new double[] { 6.7, 3.1, 5.6, 2.4, 1, 0, 0 };
			normalizedData[141] = new double[] { 6.9, 3.1, 5.1, 2.3, 1, 0, 0 };
			normalizedData[142] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
			normalizedData[143] = new double[] { 6.8, 3.2, 5.9, 2.3, 1, 0, 0 };
			normalizedData[144] = new double[] { 6.7, 3.3, 5.7, 2.5, 1, 0, 0 };
			normalizedData[145] = new double[] { 6.7, 3.0, 5.2, 2.3, 1, 0, 0 };
			normalizedData[146] = new double[] { 6.3, 2.5, 5.0, 1.9, 1, 0, 0 };
			normalizedData[147] = new double[] { 6.5, 3.0, 5.2, 2.0, 1, 0, 0 };
			normalizedData[148] = new double[] { 6.2, 3.4, 5.4, 2.3, 1, 0, 0 };
			normalizedData[149] = new double[] { 5.9, 3.0, 5.1, 1.8, 1, 0, 0 };

			var trainData = normalizedData.Take((int) (normalizedData.Length * 0.8)).ToArray();

			#endregion

			#region Test Data

			var testData = normalizedData.Skip((int) (normalizedData.Length * 0.8)).ToArray();

			#endregion

			Console.WriteLine("Number of input nodes: ");
			int numInput = int.Parse(Console.ReadLine());

			Console.WriteLine("Number of hidden nodes: ");
			int numHidden = int.Parse(Console.ReadLine());

			Console.WriteLine("Number of output nodes: ");
			int numOutput = int.Parse(Console.ReadLine());

			Console.WriteLine("Creating a neural netwotk with {0} input node(s), {1} hidden node(s) and {2} output node(s)", numInput, numHidden, numOutput);
			var configuration = new NeuralNetworkConfiguration
			{
				ActivationFunction = ActivationFunctionEnum.HyperTan,
				NumberOfHiddenNodes = numHidden,
				NumberOfInputNodes = numInput,
				NumberOfOutputNodes = numOutput
			};


			var nn = new BackPropagationNeuralNetwork(configuration);

			int maxEpochs = int.Parse(ConfigurationManager.AppSettings["MaxEpochs"]);
			double learnRate = double.Parse(ConfigurationManager.AppSettings["LearnRate"]);
			double momentum = double.Parse(ConfigurationManager.AppSettings["Momentum"]);

			Console.WriteLine("Setting maxEpochs = " + maxEpochs);
			Console.WriteLine("Setting learnRate = " + learnRate);
			Console.WriteLine("Setting momentum  = " + momentum);

			Console.WriteLine("\nBeginning training using back-propagation\n");
			var trainingData = new NeuralNetworkTrainingData
			{
				LearnRate = learnRate,
				MaxEpochs = maxEpochs,
				Momentum = momentum,
				TrainingData = trainData
			};
			nn.Train(trainingData);
			Console.WriteLine("Training complete");

			var bestWeights = nn.GetWeights();
			Console.WriteLine("Final neural network weights and bias values:");
			ShowVector(bestWeights, 10, 3, true);
			nn.SetWeightsAndBiases(bestWeights);
			var trainAcc = nn.Accuracy(trainData);
			Console.WriteLine("\nAccuracy on training data = " +
							  trainAcc.ToString("F4"));

			var testAcc = nn.Accuracy(testData);
			Console.WriteLine("Accuracy on test data = " +
							  testAcc.ToString("F4"));

			Console.WriteLine("\nEnd neural network back propagation demo\n");

			//Console.WriteLine("Start particle swarm optimization demo");

			//Console.WriteLine("Number of input nodes: ");
			//int inputNum = int.Parse(Console.ReadLine());

			//Console.WriteLine("Number of hidden nodes: ");
			//int hiddenNum = int.Parse(Console.ReadLine());

			//Console.WriteLine("Number of output nodes: ");
			//int outputNum = int.Parse(Console.ReadLine());

			//Console.WriteLine("Creating a particle swarm optimized neural netwotk with {0} input node(s), {1} hidden node(s) and {2} output node(s)", inputNum, hiddenNum, outputNum);
			//var nn_conf = new NeuralNetworkConfiguration
			//{
			//	ActivationFunction = ActivationFunctionEnum.HyperTan,
			//	NumberOfHiddenNodes = hiddenNum,
			//	NumberOfInputNodes = inputNum,
			//	NumberOfOutputNodes = outputNum
			//};

			//Console.WriteLine("Max number of epochs:");
			//var epochs = int.Parse(Console.ReadLine());

			//Console.WriteLine("Number of particles:");
			//var particles = int.Parse(Console.ReadLine());

			//Console.WriteLine("Dimension min & max:");
			//var dimensionMinMax = double.Parse(Console.ReadLine());

			//Console.WriteLine("Exit error condition:");
			//var exitError = double.Parse(Console.ReadLine());

			//var pso_conf = new ParticleSwarmConfiguration
			//{
			//	MaxEpochs = epochs,
			//	NeuralNetwork = new FeedForwardNeuralNetwork(nn_conf),
			//	dataVectors = trainData,
			//	NumbeOfParticles = particles,
			//	Dimensions = (inputNum * hiddenNum) + (hiddenNum * outputNum) + outputNum + hiddenNum,
			//	DimensionMaximum = dimensionMinMax,
			//	DimensionMinimum = dimensionMinMax * (-1),
			//	ExitError = exitError,
			//	configuration = nn_conf
			//};

			//var psnn = new ParticleSwarmNeuralNetwork(nn_conf, pso_conf);

			//psnn.Train(trainingData);

			ShowVector(nn.GetWeights(), 10, 3, true);

			Console.WriteLine("Accuracy on test data = " + nn.Accuracy(testData).ToString("F4"));

			Console.WriteLine("End Neural network demo.");
			Console.ReadLine();

		} // Main

		private static void ShowVector (double[] vector, int valsPerRow, int decimals, bool newLine)
		{
			for (var i = 0; i < vector.Length; ++i)
			{
				if (i % valsPerRow == 0)
					Console.WriteLine("");
				Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
			}
			if (newLine)
				Console.WriteLine("");
		}
	}

}
