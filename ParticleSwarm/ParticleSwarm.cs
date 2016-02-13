using System;
using System.Threading.Tasks;
using Common;

namespace ParticleSwarm
{
	public class ParticleSwarm
	{
		private readonly ParticleSwarmConfiguration _configuration;
		private readonly Random _random;

		private readonly Particle[] _swarm;
		private readonly int _vectorSize;

		public double[] BestGlobalPosition
		{
			get;
			set;
		}
		public double BestGlobalError
		{
			get;
			set;
		}

		public double InertiaWeight
		{
			get;
			set;
		}
		public double CognitiveWeight
		{
			get
			{
				return 1.49445;
			}
		}
		public double SocialWeight
		{
			get
			{
				return 1.49445;
			}
		}

		public double CognitiveRandomizer1
		{
			get;
			set;
		}
		public double CognitiveRandomizer2
		{
			get;
			set;
		}

		public double ParticleDeathProbability
		{
			get;
			set;
		}

		public ParticleSwarm (ParticleSwarmConfiguration configuration)
		{
			_random = new Random(0);
			_configuration = configuration;
			var weights = _configuration.NeuralNetwork.GetWeights();
			_vectorSize = weights.Length;
			_swarm = new Particle[configuration.NumbeOfParticles];
			ParticleDeathProbability = 0.5;
			InertiaWeight = 0.729;
		}

		public double[] Swarm ()
		{
			BestGlobalPosition = new double[_vectorSize];
			BestGlobalError = double.MaxValue;

			InitializeSwarm(_configuration.configuration, _configuration.dataVectors);
			StepTroughSwarm(_configuration.configuration, _configuration.dataVectors);

			return BestGlobalPosition;
		}

		private void StepTroughSwarm (NeuralNetworkConfiguration configuration, double[][] dataVectors)
		{
			var epoch = 0;

			var randomizedParticles = NeuralNetworkHelpers.Randomize(_swarm.Length);

			while (epoch < _configuration.MaxEpochs || BestGlobalError <= _configuration.ExitError)
			{
				if (epoch % 100 == 0)
					Console.WriteLine("Epoch {0}. Best error = {1}", epoch, BestGlobalError);

				for (var i = 0; i < _swarm.Length; i++)
				{
					var particle = _swarm[randomizedParticles[i]];

					UpdatePositionAndVelocity(particle);
					UpdateParticleError(configuration, dataVectors, particle);
					LiveOrDie(particle, configuration, dataVectors);
				}
				epoch++;
			}
		}

		private void UpdateParticleError (NeuralNetworkConfiguration configuration, double[][] dataVectors, Particle particle)
		{
			var newError = Error(configuration, dataVectors, particle.Properties.Position);
			particle.ParticleError.Error = newError;

			if (newError < particle.ParticleError.BestError)
			{
				var bestPosition = particle.Properties.BestPosition;
				Array.Copy(particle.Properties.Position, particle.Properties.BestPosition, bestPosition.Length);

				particle.ParticleError.BestError = newError;
			}

			if (!(newError < BestGlobalError))
				return;

			var position = particle.Properties.Position;

			Array.Copy(particle.Properties.Position, BestGlobalPosition, position.Length);
			BestGlobalError = newError;
		}

		private void UpdatePositionAndVelocity (Particle particle)
		{
			UpdateVelocity(particle);
			UpdatePosition(particle);
		}

		private void UpdatePosition(Particle particle)
		{
			var newPosition = new double[_vectorSize];

			Parallel.For(0, newPosition.Length, i =>
			{
				newPosition[i] = particle.Properties.Position[i] + particle.Properties.Velocity[i];

				if (newPosition[i] < _configuration.DimensionMinimum)
				{
					newPosition[i] = _configuration.DimensionMinimum;
				}

				if (newPosition[i] > _configuration.DimensionMaximum)
				{
					newPosition[i] = _configuration.DimensionMaximum;
				}
			});

			Array.Copy(newPosition, particle.Properties.Position, newPosition.Length);
		}

		private void UpdateVelocity (Particle particle)
		{
			var newVelocity = new double[_vectorSize];

			Parallel.For(0, newVelocity.Length, i =>
			{
				CognitiveRandomizer1 = _random.NextDouble();
				CognitiveRandomizer2 = _random.NextDouble();

				newVelocity[i] = (InertiaWeight * particle.Properties.Velocity[i]) +
								 (CognitiveWeight * CognitiveRandomizer1 *
								  (particle.Properties.BestPosition[i] - particle.Properties.Position[i])) +
								 (SocialWeight * CognitiveRandomizer2 * (BestGlobalPosition[i] - particle.Properties.Position[i]));
			});

			Array.Copy(newVelocity, particle.Properties.Velocity, newVelocity.Length);
		}

		private void LiveOrDie (Particle particle, NeuralNetworkConfiguration configuration, double[][] dataVectors)
		{
			var currentLifeProbability = _random.NextDouble();

			if (!(currentLifeProbability < ParticleDeathProbability))
				return;

			var position = particle.Properties.Position;
			for (var i = 0; i < position.Length; i++)
				particle.Properties.Position[i] = (_configuration.DimensionMaximum - _configuration.DimensionMinimum) * _random.NextDouble() + _configuration.DimensionMinimum;

			particle.ParticleError.Error = Error(configuration, dataVectors, particle.Properties.Position);
			particle.ParticleError.BestError = particle.ParticleError.Error;

			Array.Copy(particle.Properties.Position, particle.Properties.BestPosition,position.Length);

			if (!(particle.ParticleError.Error < BestGlobalError))
				return;

			BestGlobalError = particle.ParticleError.Error;
			var particlePosition = particle.Properties.Position;
			Array.Copy(particle.Properties.Position, BestGlobalPosition, particlePosition.Length);
		}

		private void InitializeSwarm (NeuralNetworkConfiguration configuration, double[][] dataVectors)
		{
			for (int i = 0; i < _swarm.Length; i++)
			{
				CreateParticleForSwarm(configuration, dataVectors, i);

				var localParticleError = _swarm[i].ParticleError;

				if (localParticleError.Error < BestGlobalError)
				{
					var particlePosition = _swarm[i].Properties;
					BestGlobalError = localParticleError.Error;
					Array.Copy(particlePosition.Position, BestGlobalPosition, particlePosition.Position.Length);
				}
			}
		}

		private void CreateParticleForSwarm(NeuralNetworkConfiguration configuration, double[][] dataVectors, int i)
		{
			var properties = new ParticleProperties(_vectorSize);
			var randomPosition = InitializeRandomPosition();

			var error = Error(configuration, dataVectors, randomPosition);

			var particleError = new ParticleError
			{
				Error = error,
				BestError = error
			};

			var randomVelocity = InitializeRandomVelocity();

			Array.Copy(randomPosition, properties.Position, randomPosition.Length);
			Array.Copy(randomPosition, properties.BestPosition, randomPosition.Length);
			Array.Copy(randomVelocity, properties.Velocity, randomVelocity.Length);

			_swarm[i] = new Particle(particleError, properties);
		}

		private double[] InitializeRandomVelocity ()
		{
			var randomVelocity = new double[_vectorSize];

			for (var i = 0; i < randomVelocity.Length; i++)
			{
				randomVelocity[i] = ((_configuration.DimensionMaximum * 0.1) - (_configuration.DimensionMinimum * 0.1)) * _random.NextDouble() + (_configuration.DimensionMinimum * 0.1);
			}
			return randomVelocity;
		}

		private double[] InitializeRandomPosition ()
		{
			var randomPosition = new double[_vectorSize];

			for (var i = 0; i < randomPosition.Length; i++)
			{
				randomPosition[i] = (_configuration.DimensionMaximum - _configuration.DimensionMinimum) * _random.NextDouble() + _configuration.DimensionMinimum;
			}

			return randomPosition;
		}

		public double Error (NeuralNetworkConfiguration configuration, double[][] trainDataVector, double[] position)
		{
			var squaredError = 0.0;
			_configuration.NeuralNetwork.SetWeightsAndBiases(position);
			var inputVector = new double[configuration.NumberOfInputNodes];
			var expectedResult = new double[configuration.NumberOfOutputNodes];

			foreach (var vector in trainDataVector)
			{
				Array.Copy(vector, inputVector, configuration.NumberOfInputNodes);
				Array.Copy(vector, configuration.NumberOfInputNodes, expectedResult, 0, configuration.NumberOfOutputNodes);

				var output = _configuration.NeuralNetwork.ComputeOutputs(inputVector);

				for (var i = 0; i < output.Length; i++)
				{
					squaredError += ((output[i] - expectedResult[i]) * (output[i] - expectedResult[i]));
				}
			}

			return squaredError / trainDataVector.Length;
		}

		public double[] Train ()
		{
			return Swarm();
		}
	}
}