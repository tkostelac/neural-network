using System;
using System.Collections.Generic;
using Common;
using Interfaces;

namespace NeuralNetwork
{
	public class FeedForwardNeuralNetwork : INeuralNetwork
	{
		private readonly NeuralNetworkConfiguration _configuration;
		private readonly int _numberOfWeights;

		private readonly double[] _inputs;
		private readonly double[][] _hiddenInputWeights;
		private readonly double[] _hiddenBiases;
		private readonly double[] _hiddenOutputs;
		private readonly double[][] _hiddenOutputWeights;
		private readonly double[] _outputs;
		private readonly double[] _outputBiases;

		public FeedForwardNeuralNetwork (NeuralNetworkConfiguration configuration)
		{
			_configuration = configuration;

			_hiddenBiases = new double[configuration.NumberOfHiddenNodes];
			_hiddenOutputs = new double[configuration.NumberOfHiddenNodes];

			_outputBiases = new double[configuration.NumberOfOutputNodes];
			_outputs = new double[configuration.NumberOfOutputNodes];
			_inputs = new double[configuration.NumberOfInputNodes];
			_hiddenOutputWeights = NeuralNetworkHelpers.CreateMatrix(configuration.NumberOfHiddenNodes, configuration.NumberOfOutputNodes);

			_hiddenInputWeights = NeuralNetworkHelpers.CreateMatrix(configuration.NumberOfInputNodes, configuration.NumberOfHiddenNodes);

			_numberOfWeights = DetermineNumberOfWeightsAndBiases();
		}



		public void SetWeightsAndBiases (double[] weights)
		{
			if (weights.Length != _numberOfWeights)
				throw new Exception("Bad number of weights and / or biases in array.");

			var weightIndex = 0;

			SetHiddenInputWeights(weights, ref weightIndex);
			SetHiddenBiases(weights, ref weightIndex);
			SetHiddenOutputWeights(weights, ref weightIndex);
			SetOutoutBiases(weights, ref weightIndex);
		}

		public double[] GetWeights ()
		{
			var result = new double[_numberOfWeights];
			var weightIndex = 0;

			AddHiddenInputWeights(ref result, ref weightIndex);
			AddHiddenBiases(ref result, ref weightIndex);
			AddHiddenOutputWeights(ref result, ref weightIndex);
			AddOutputBiases(ref result, ref weightIndex);

			return result;
		}

		public void Train (NeuralNetworkTrainingData data)
		{
			throw new System.NotImplementedException();
		}

		public double Accuracy (double[][] testData)
		{
			var correct = 0;
			var wrong = 0;

			var inputVector = new double[_configuration.NumberOfInputNodes];
			var expectedValue = new double[_configuration.NumberOfOutputNodes];

			for (var i = 0; i < testData.Length; i++)
			{
				Array.Copy(testData[i], inputVector, _configuration.NumberOfInputNodes);
				Array.Copy(testData[i], _configuration.NumberOfInputNodes, expectedValue, 0, _configuration.NumberOfOutputNodes);

				var resultVector = ComputeOutputs(inputVector);
				var indexOfLargestValue = NeuralNetworkHelpers.MaxValueIndex(resultVector);

				if (expectedValue[indexOfLargestValue].Equals(1.0))
					correct++;
				else
					wrong++;
			}

			return (correct * 1.0) / (correct + wrong);
		}

		public double[] ComputeOutputs (double[] inputVector)
		{
			ValidateInputVector(inputVector);

			var hiddenSum = new double[_configuration.NumberOfHiddenNodes];
			var outputSum = new double[_configuration.NumberOfOutputNodes];

			CopyInputVetorToInputs(inputVector);
			AddWeightsToHiddenInputs(ref hiddenSum);
			ActivateHiddenNodes(hiddenSum);
			AddWeightsToHiddenOutouts(ref outputSum);

			var softOutputs = NeuralNetworkHelpers.SoftMax(outputSum);
			Array.Copy(softOutputs, _outputs, softOutputs.Length);

			var result = new double[_configuration.NumberOfOutputNodes];
			Array.Copy(_outputs, result, result.Length);

			return result;
		}

		public int DetermineNumberOfWeightsAndBiases ()
		{
			var numberOfWeightsBetweenInputAndHiddenNodes = _configuration.NumberOfInputNodes * _configuration.NumberOfHiddenNodes;
			var numberOfWeightsBetweenHiddenAndOutputNodes = _configuration.NumberOfHiddenNodes * _configuration.NumberOfOutputNodes;

			return numberOfWeightsBetweenInputAndHiddenNodes +
					numberOfWeightsBetweenHiddenAndOutputNodes +
					_configuration.NumberOfHiddenNodes +
					_configuration.NumberOfOutputNodes;
		}

		private void ValidateInputVector (ICollection<double> inputVector)
		{
			if (inputVector.Count != _configuration.NumberOfInputNodes)
			{
				throw new Exception(
					string.Format("Input vector doesn't match the number of input nodes. Input vector size = {0}, expected = {1}",
						inputVector.Count, _configuration.NumberOfInputNodes));
			}
		}

		private void AddWeightsToHiddenOutouts (ref double[] outputSum)
		{
			for (var j = 0; j < _configuration.NumberOfOutputNodes; j++)
			{
				for (var i = 0; i < _configuration.NumberOfHiddenNodes; i++)
				{
					outputSum[j] += _hiddenOutputs[i] * _hiddenOutputWeights[i][j];
				}
				outputSum[j] += _outputBiases[j];
			}
		}

		private void ActivateHiddenNodes (double[] hiddenSum)
		{
			switch (_configuration.ActivationFunction)
			{
				case ActivationFunctionEnum.HyperTan:
				ApplyHyperTanActivation(hiddenSum);
				break;

				case ActivationFunctionEnum.Sigmoid:
				ApplySigmoidActivation(hiddenSum);
				break;

				case ActivationFunctionEnum.Gaussian:
				ApplyGaussianActivation(hiddenSum);
				break;
			}
		}

		private void ApplyGaussianActivation (double[] hiddenSum)
		{
			for (var i = 0; i < _configuration.NumberOfHiddenNodes; i++)
			{
				_hiddenOutputs[i] = NeuralNetworkHelpers.Gaussian(hiddenSum[i]);
			}
		}

		private void ApplySigmoidActivation (double[] hiddenSum)
		{
			for (var i = 0; i < _configuration.NumberOfHiddenNodes; i++)
			{
				_hiddenOutputs[i] = NeuralNetworkHelpers.Sigmoid(hiddenSum[i]);
			}
		}

		private void ApplyHyperTanActivation (double[] hiddenSum)
		{
			for (var i = 0; i < _configuration.NumberOfHiddenNodes; i++)
			{
				_hiddenOutputs[i] = NeuralNetworkHelpers.HyperTan(hiddenSum[i]);
			}
		}

		private void AddWeightsToHiddenInputs (ref double[] hiddenSum)
		{
			for (var j = 0; j < _configuration.NumberOfHiddenNodes; j++)
			{
				for (var i = 0; i < _configuration.NumberOfInputNodes; i++)
				{
					hiddenSum[j] += _inputs[i] * _hiddenInputWeights[i][j];
				}

				hiddenSum[j] += _hiddenBiases[j];
			}
		}

		private void CopyInputVetorToInputs (double[] inputVector)
		{
			for (var i = 0; i < inputVector.Length; i++)
			{
				_inputs[i] = inputVector[i];
			}
		}

		private void AddOutputBiases (ref double[] result, ref int weightIndex)
		{
			foreach (var bias in _outputBiases)
			{
				result[weightIndex++] = bias;
			}
		}

		private void AddHiddenOutputWeights (ref double[] result, ref int weightIndex)
		{
			foreach (var hiddenOutputWeight in _hiddenOutputWeights)
			{
				for (var j = 0; j < _hiddenOutputWeights[0].Length; j++)
				{
					result[weightIndex++] = hiddenOutputWeight[j];
				}
			}
		}

		private void AddHiddenBiases (ref double[] result, ref int weightIndex)
		{
			foreach (var bias in _hiddenBiases)
			{
				result[weightIndex++] = bias;
			}
		}

		private void AddHiddenInputWeights (ref double[] result, ref int weightIndex)
		{
			foreach (double[] inputWeight in _hiddenInputWeights)
			{
				for (var j = 0; j < _hiddenInputWeights[0].Length; j++)
				{
					result[weightIndex++] = inputWeight[j];
				}
			}
		}

		private void SetOutoutBiases (IList<double> weights, ref int weightIndex)
		{
			for (var i = 0; i < _configuration.NumberOfOutputNodes; i++)
			{
				_outputBiases[i] = weights[weightIndex++];
			}
		}

		private void SetHiddenOutputWeights (IList<double> weights, ref int weightIndex)
		{
			for (var i = 0; i < _configuration.NumberOfHiddenNodes; i++)
			{
				for (var j = 0; j < _configuration.NumberOfOutputNodes; j++)
				{
					_hiddenOutputWeights[i][j] = weights[weightIndex++];
				}
			}
		}

		private void SetHiddenBiases (IList<double> weights, ref int weightIndex)
		{
			for (var i = 0; i < _configuration.NumberOfHiddenNodes; i++)
			{
				_hiddenBiases[i] = weights[weightIndex++];
			}
		}

		private void SetHiddenInputWeights (IList<double> weights, ref int weightIndex)
		{
			for (var i = 0; i < _configuration.NumberOfInputNodes; i++)
			{
				for (var j = 0; j < _configuration.NumberOfHiddenNodes; j++)
				{
					_hiddenInputWeights[i][j] = weights[weightIndex++];
				}
			}
		}


	}
}