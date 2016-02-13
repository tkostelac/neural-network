using Common;
using Interfaces;

namespace ParticleSwarm
{
	public class ParticleSwarmConfiguration
	{
		public int Dimensions { get; set; }
		public int NumbeOfParticles { get; set; }
		public double DimensionMinimum { get; set; }
		public double DimensionMaximum { get; set; }
		public double ExitError { get; set; }
		public int MaxEpochs { get; set; }
		public INeuralNetwork NeuralNetwork { get; set; }
		public NeuralNetworkConfiguration configuration { get; set; }
		public double[][] dataVectors { get; set; }
	}
}