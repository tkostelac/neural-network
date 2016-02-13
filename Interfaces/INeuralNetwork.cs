using Common;

namespace Interfaces
{
	public interface INeuralNetwork
	{
		void SetWeightsAndBiases(double[] weights);
		double[] GetWeights();
		void Train(NeuralNetworkTrainingData data);
		double Accuracy(double[][] testData);
		double[] ComputeOutputs(double[] inputVector);
	}
}