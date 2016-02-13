using System;

namespace ParticleSwarm
{
	public class Particle
	{
		private ParticleProperties _properties;
		private ParticleError _particleError;

		public ParticleProperties Properties
		{
			get { return _properties; }
			set { _properties = value; }
		}

		public ParticleError ParticleError
		{
			get { return _particleError; }
			set { _particleError = value; }
		}

		public Particle(ParticleError particleError, ParticleProperties properties)
		{
			_properties = properties;
			_particleError = particleError;
		}
	}
}