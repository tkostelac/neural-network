namespace Common
{
	public class NetworkTrainingParameters
	{
		public double LeaningRate { get; set; }
		public double Momentum { get; set; }
		public double[] ExpectedResult { get; set; }
		public double[] OutputGradients { get; set; }
		public double[] HiddenGradients { get; set; }

		public NetworkTrainingParameters(int expectedResultLength, int hiddenGradientLength)
		{
			ExpectedResult = new double[expectedResultLength];
			OutputGradients = new double[expectedResultLength];
			HiddenGradients = new double[hiddenGradientLength];
		}
	}
}