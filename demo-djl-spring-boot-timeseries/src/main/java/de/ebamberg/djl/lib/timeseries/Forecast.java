package de.ebamberg.djl.lib.timeseries;

public class Forecast {

	private float[] forecastValue;

	public Forecast(float[] forecastValue) {
		super();
		this.forecastValue = forecastValue;
	}

	public float[] getForecastValue() {
		return forecastValue;
	}
	
	
	
}
