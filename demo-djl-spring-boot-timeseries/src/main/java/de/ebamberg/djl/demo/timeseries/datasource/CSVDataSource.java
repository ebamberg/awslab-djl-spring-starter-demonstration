package de.ebamberg.djl.demo.timeseries.datasource;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.springframework.stereotype.Component;

@Component
public class CSVDataSource {

    public float[][] read(InputStream inputStream) throws IOException {
		List<Float> datapoints=new ArrayList<>(1000);
		try (	Reader in = new InputStreamReader( inputStream ) ) {
		    CSVFormat.DEFAULT
			    .withHeader(new String[]{ "Month", "Passengers"})
			    .withFirstRecordAsHeader()
			    .parse(in)
			    .forEach(record -> datapoints.add(Float.valueOf(record.get("Passengers"))) );
		}
		return convertToPrimitiveFloats(datapoints);
    }
    
    private float[][] convertToPrimitiveFloats(List<Float> values) {
	    int length = values.size();
	    float[][] result = new float[length][1];
	    for (int i = 0; i < length; i++) {
	      result[i][0] = values.get(i).floatValue();
	    }
	    return result;
	  }

	
}
