package de.ebamberg.djl.demo.timeseries.repository;

import java.io.IOException;
import java.nio.file.Path;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.nn.BlockFactory;

@Controller
public class ModelRepository {

	@Value("${repository.path}")
	private String modelPath;
	
	@Autowired
	private BlockFactory blockFactory;
	
	public void store(Model model) throws IOException {
		model.save(Path.of(modelPath), model.getName());
	}
	


}
