using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Common;
using Interfaces;

namespace NeuralNetwork
{
	public class BackPropagationNeuralNetwork : IDisposable, INeuralNetwork
	{
		private readonly int _numberOfInputNodes;
		private readonly int _numberOfHiddenNodes;
		private readonly int _numberOfOutputNodes;

		private readonly int _numberOfWeights;

		private readonly double[] _inputs;
		private readonly double[][] _hiddenInputWeights;
		private readonly double[] _hiddenBiases;
		private readonly double[] _hiddenOutputs;
		private readonly double[][] _hiddenOutputWeights;
		private readonly double[] _outputs;
		private readonly double[] _outputBiases;
		
		private readonly double[][] _previousHiddenInputWeightDelta;
		private readonly double[][] _previousHiddenOutputWeightDelta;

		private readonly double[] _previousHiddenBiasesDelta;
		private readonly double[] _previousOutputBiasesDelta;

		private readonly ActivationFunctionEnum _function;

// ReSharper disable once TooManyDependencies
		public BackPropagationNeuralNetwork (NeuralNetworkConfiguration configuration)
		{
			_numberOfInputNodes = configuration.NumberOfInputNodes;
			_numberOfHiddenNodes = configuration.NumberOfHiddenNodes;
			_numberOfOutputNodes = configuration.NumberOfOutputNodes;

			_hiddenBiases = new double[configuration.NumberOfHiddenNodes];
			_hiddenOutputs = new double[configuration.NumberOfHiddenNodes];

			_outputBiases = new double[configuration.NumberOfOutputNodes];
			_outputs = new double[configuration.NumberOfOutputNodes];
			_inputs = new double[configuration.NumberOfInputNodes];
			_function = configuration.ActivationFunction;

			_previousHiddenBiasesDelta = new double[configuration.NumberOfHiddenNodes];
			_previousOutputBiasesDelta = new double[configuration.NumberOfOutputNodes];

			_hiddenOutputWeights = NeuralNetworkHelpers.CreateMatrix(configuration.NumberOfHiddenNodes, configuration.NumberOfOutputNodes);

			_hiddenInputWeights = NeuralNetworkHelpers.CreateMatrix(configuration.NumberOfInputNodes, configuration.NumberOfHiddenNodes);

			_previousHiddenInputWeightDelta = NeuralNetworkHelpers.CreateMatrix(configuration.NumberOfInputNodes, configuration.NumberOfHiddenNodes);

			_previousHiddenOutputWeightDelta = NeuralNetworkHelpers.CreateMatrix(configuration.NumberOfHiddenNodes, configuration.NumberOfOutputNodes);

			_numberOfWeights = DetermineNumberOfWeightsAndBiases();

			InitializeWeights();
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
			var trainingParameters = new NetworkTrainingParameters(_numberOfOutputNodes, _numberOfHiddenNodes)
			{
				LeaningRate = data.LearnRate,
				Momentum = data.Momentum
			};

			var inputVector = new double[_numberOfInputNodes];

			for (var i = 0; i < data.MaxEpochs; i++)
			{
				var sequence = NeuralNetworkHelpers.Randomize(data.TrainingData.Length);

				foreach(var index in sequence)
				{
					Array.Copy(data.TrainingData[index], inputVector, _numberOfInputNodes);
					Array.Copy(data.TrainingData[index], _numberOfInputNodes, trainingParameters.ExpectedResult, 0, _numberOfOutputNodes);

					ComputeOutputs(inputVector);
					TrainNetwork(ref trainingParameters);
				}
			}
		}

		public double Accuracy (double[][] testData)
		{
			var correct = 0;
			var wrong = 0;

			var inputVector = new double[_numberOfInputNodes];
			var expectedValue = new double[_numberOfOutputNodes];

			for(var i = 0; i< testData.Length; i++)
			{
				Array.Copy(testData[i], inputVector, _numberOfInputNodes);
				Array.Copy(testData[i], _numberOfInputNodes, expectedValue, 0, _numberOfOutputNodes);

				var resultVector = ComputeOutputs(inputVector);
				var indexOfLargestValue = NeuralNetworkHelpers.MaxValueIndex(resultVector);

				if (expectedValue[indexOfLargestValue].Equals(1.0))
					correct++;
				else
					wrong++;
			}

			return (correct * 1.0) / (correct + wrong);
		}

		private void TrainNetwork (ref NetworkTrainingParameters parameters)
		{
			ComputeOutputGradients(parameters.ExpectedResult, parameters.OutputGradients);
			ComputeHiddenGradients(parameters.OutputGradients, parameters.HiddenGradients);
			UpdateHiddenWeights(parameters.LeaningRate, parameters.Momentum, parameters.HiddenGradients);
			UpdateHiddenBiases(parameters.LeaningRate, parameters.Momentum, parameters.HiddenGradients);
			UpdateHiddenOutputWeights(parameters.LeaningRate, parameters.Momentum, parameters.OutputGradients);
			UpdateOutputBiases(parameters.LeaningRate, parameters.Momentum, parameters.OutputGradients);
		}

		private void UpdateOutputBiases (double learnRate, double momentum, IList<double> outputGradients)
		{
			Parallel.For(0, _numberOfOutputNodes, i =>
			{
				var delta = learnRate * outputGradients[i] * 1.0;
				_outputBiases[i] += delta;
				_outputBiases[i] += momentum * _previousOutputBiasesDelta[i];
				_previousOutputBiasesDelta[i] = delta;
			});
		}

		private void UpdateHiddenOutputWeights (double learnRate, double momentum, IList<double> outputGradients)
		{
			Parallel.For(0, _numberOfHiddenNodes, i =>
				Parallel.For(0, _numberOfOutputNodes, j =>
				{
					var delta = learnRate * outputGradients[j] * _hiddenOutputs[i];
					_hiddenOutputWeights[i][j] += delta;
					_hiddenOutputWeights[i][j] += momentum * _previousHiddenOutputWeightDelta[i][j];
					_previousHiddenOutputWeightDelta[i][j] = delta;
				})
			);
		}

		private void UpdateHiddenBiases (double learnRate, double momentum, IList<double> hiddenGradients)
		{
			Parallel.For(0, _numberOfHiddenNodes, i =>
			{
				var delta = learnRate * hiddenGradients[i];
				_hiddenBiases[i] += delta;
				_hiddenBiases[i] += momentum * _previousHiddenBiasesDelta[i];
				_previousHiddenBiasesDelta[i] = delta;
			});
		}

		private void UpdateHiddenWeights (double learnRate, double momentum, IList<double> hiddenGradients)
		{
			Parallel.For(0, _numberOfInputNodes, i =>
				Parallel.For(0, _numberOfHiddenNodes, j =>
				{
					var delta = learnRate * hiddenGradients[j] * _inputs[i];
					_hiddenInputWeights[i][j] += delta;
					_hiddenInputWeights[i][j] += momentum * _previousHiddenInputWeightDelta[i][j];
					_previousHiddenInputWeightDelta[i][j] = delta;
				})
			);
		}

		private void ComputeHiddenGradients (IList<double> outputGradients, IList<double> hiddenGradients)
		{
			Parallel.For(0, _numberOfHiddenNodes, i =>
			{
				var derivative = (1 - _hiddenOutputs[i]) * (1 + _hiddenOutputs[i]);

				var sum = 0.0;

				Parallel.For(0, _numberOfOutputNodes, j =>
				{
					sum += outputGradients[j] * _hiddenOutputWeights[i][j];
				});

				hiddenGradients[i] = derivative * sum;

			});
		}

		private void ComputeOutputGradients (IList<double> expectedResult, IList<double> outputGradients)
		{
			Parallel.For(0, _numberOfOutputNodes, i =>
			{
				var derivative = (1 - _outputs[i]) * _outputs[i];
				outputGradients[i] = derivative * (expectedResult[i] - _outputs[i]);

			});
		}

		// ReSharper disable once UnusedMethodReturnValue.Local
		public double[] ComputeOutputs (double[] inputVector)
		{
			ValidateInputVector(inputVector);

			var hiddenSum = new double[_numberOfHiddenNodes];
			var outputSum = new double[_numberOfOutputNodes];

			CopyInputVetorToInputs(inputVector);
			AddWeightsToHiddenInputs(ref hiddenSum);
			ActivateHiddenNodes(hiddenSum);
			AddWeightsToHiddenOutouts(ref outputSum);

			var softOutputs = NeuralNetworkHelpers.SoftMax(outputSum);
			Array.Copy(softOutputs, _outputs, softOutputs.Length);

			var result = new double[_numberOfOutputNodes];
			Array.Copy(_outputs, result, result.Length);

			return result;
		}

		private void ValidateInputVector (ICollection<double> inputVector)
		{
			if (inputVector.Count != _numberOfInputNodes)
			{
				throw new Exception(
					string.Format("Input vector doesn't match the number of input nodes. Input vector size = {0}, expected = {1}",
						inputVector.Count, _numberOfInputNodes));
			}
		}

		private void AddWeightsToHiddenOutouts (ref double[] outputSum)
		{
			for (var j = 0; j < _numberOfOutputNodes; j++)
			{
				for (var i = 0; i < _numberOfHiddenNodes; i++)
				{
					outputSum[j] += _hiddenOutputs[i] * _hiddenOutputWeights[i][j];
				}
				outputSum[j] += _outputBiases[j];
			}
		}

		private void ActivateHiddenNodes (double[] hiddenSum)
		{
			switch (_function)
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
			for (var i = 0; i < _numberOfHiddenNodes; i++)
			{
				_hiddenOutputs[i] = NeuralNetworkHelpers.Gaussian(hiddenSum[i]);
			}
		}

		private void ApplySigmoidActivation (double[] hiddenSum)
		{
			for (var i = 0; i < _numberOfHiddenNodes; i++)
			{
				_hiddenOutputs[i] = NeuralNetworkHelpers.Sigmoid(hiddenSum[i]);
			}
		}

		private void ApplyHyperTanActivation (double[] hiddenSum)
		{
			for (var i = 0; i < _numberOfHiddenNodes; i++)
			{
				_hiddenOutputs[i] = NeuralNetworkHelpers.HyperTan(hiddenSum[i]);
			}
		}

		private void AddWeightsToHiddenInputs (ref double[] hiddenSum)
		{
			for (var j = 0; j < _numberOfHiddenNodes; j++)
			{
				for (var i = 0; i < _numberOfInputNodes; i++)
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

		private void InitializeWeights ()
		{
			var random = new Random(0);
			var initialWeightsAndBiases = new double[_numberOfWeights];

			const double low = -0.01;
			const double high = 0.01;

			for (var i = 0; i < _numberOfWeights; i++)
			{
				initialWeightsAndBiases[i] = (high - low) * random.NextDouble() + low;
			}

			SetWeightsAndBiases(initialWeightsAndBiases);
		}

		private void SetOutoutBiases (IList<double> weights, ref int weightIndex)
		{
			for (var i = 0; i < _numberOfOutputNodes; i++)
			{
				_outputBiases[i] = weights[weightIndex++];
			}
		}

		private void SetHiddenOutputWeights (IList<double> weights, ref int weightIndex)
		{
			for (var i = 0; i < _numberOfHiddenNodes; i++)
			{
				for (var j = 0; j < _numberOfOutputNodes; j++)
				{
					_hiddenOutputWeights[i][j] = weights[weightIndex++];
				}
			}
		}

		private void SetHiddenBiases (IList<double> weights, ref int weightIndex)
		{
			for (var i = 0; i < _numberOfHiddenNodes; i++)
			{
				_hiddenBiases[i] = weights[weightIndex++];
			}
		}

		private void SetHiddenInputWeights (IList<double> weights, ref int weightIndex)
		{
			for (var i = 0; i < _numberOfInputNodes; i++)
			{
				for (var j = 0; j < _numberOfHiddenNodes; j++)
				{
					_hiddenInputWeights[i][j] = weights[weightIndex++];
				}
			}
		}

		private int DetermineNumberOfWeightsAndBiases ()
		{
			var numberOfWeightsBetweenInputAndHiddenNodes = _numberOfInputNodes * _numberOfHiddenNodes;
			var numberOfWeightsBetweenHiddenAndOutputNodes = _numberOfHiddenNodes * _numberOfOutputNodes;

			return numberOfWeightsBetweenInputAndHiddenNodes +
					numberOfWeightsBetweenHiddenAndOutputNodes +
					_numberOfHiddenNodes +
					_numberOfOutputNodes;
		}

		public void Dispose()
		{
			GC.SuppressFinalize(this);
		}
	}
}