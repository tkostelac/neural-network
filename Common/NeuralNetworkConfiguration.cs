namespace Common
{
	public class NeuralNetworkConfiguration
	{
		public int NumberOfInputNodes { get; set; }
		public int NumberOfHiddenNodes { get; set; }
		public int NumberOfOutputNodes { get; set; }
		public ActivationFunctionEnum ActivationFunction { get; set; }
	}
}