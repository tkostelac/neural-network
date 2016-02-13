namespace ParticleSwarm
{
	public class ParticleProperties
	{
		private double[] _velocity;
		private double[] _position;
		private double[] _bestPosition;

		public double[] Velocity
		{
			get { return _velocity; }
			set { _velocity = value; }
		}

		public double[] Position
		{
			get { return _position; }
			set { _position = value; }
		}

		public double[] BestPosition
		{
			get { return _bestPosition; }
			set { _bestPosition = value; }
		}

		public ParticleProperties(int outputVectorSize)
		{
			_velocity = new double[outputVectorSize];
			_position = new double[outputVectorSize];
			_bestPosition = new double[outputVectorSize];
		}
	}
}