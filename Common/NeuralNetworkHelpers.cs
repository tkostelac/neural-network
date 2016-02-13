using System;
using System.Linq;
using System.Threading.Tasks;

namespace Common
{
	public static class NeuralNetworkHelpers
	{
		public static double[][] CreateMatrix (int rows, int columns)
		{
			var result = new double[rows][];

			for (var i = 0; i < rows; i++)
			{
				result[i] = new double[columns];
			}

			return result;
		}

		public static double[] SoftMax (double[] outputSum)
		{
			var max = outputSum.Max();

			var scale = 0.0;

			for (var i = 0; i < outputSum.Length; i++)
			{
				scale += Math.Exp(outputSum[i] - max);
			}

			var result = new double[outputSum.Length];

			for (int i = 0; i < outputSum.Length; i++)
			{
				result[i] = Math.Exp(outputSum[i] - max) / scale;
			}

			return result;
		}

		public static int MaxValueIndex (double[] vector)
		{
			var max = vector.Max();
			return Array.IndexOf(vector, max);
		}

		public static double HyperTan(double input)
		{
			if (input < -20.0)
				return -1.0;

			if (input > 20.0)
				return 1.0;

			return Math.Tanh(input);
		}

		public static double Sigmoid(double d)
		{
			throw new NotImplementedException();
		}

		public static double Gaussian(double d)
		{
			throw new NotImplementedException();
		}

		public static int[] Randomize (int sequenceLength)
		{
			var random = new Random(0);

			var sequence = new int[sequenceLength];

			Parallel.For(0, sequenceLength, i => sequence[i] = i++);
			
			Parallel.For(0, sequenceLength, i =>
			{
				var randomPosition = random.Next(i, sequence.Length);
				var temporaryRecord = sequence[randomPosition];
				sequence[randomPosition] = sequence[i];
				sequence[i] = temporaryRecord;
			});

			return sequence;
		}
	}
}