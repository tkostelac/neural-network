namespace Common
{
	public class NeuralNetworkTrainingData
	{
		public double[][] TrainingData { get; set; }
		public int MaxEpochs { get; set; }
		public double LearnRate { get; set; }
		public double Momentum { get; set; }
	}
}