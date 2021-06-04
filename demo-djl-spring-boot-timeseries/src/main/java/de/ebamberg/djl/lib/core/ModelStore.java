package de.ebamberg.djl.lib.core;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.nn.BlockFactory;

public class ModelStore implements AutoCloseable {
	
	private static Logger log=LoggerFactory.getLogger(ModelStore.class);

	private BlockFactory blockFactory;
	private Map<String,Model> modelCache=new ConcurrentHashMap<>();
	
	private String modelPath;
	

	public ModelStore(BlockFactory blockFactory) {
		super();
		this.blockFactory = blockFactory;
	}


	public Model findModel(String modelName) throws MalformedModelException, IOException {
		return modelCache.computeIfAbsent(modelName, modelname-> {
			Model model=createEmptyModel( modelName );
			try {
				load(model);
			} catch (MalformedModelException | IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return model;					
		});
	}
	
	public Model createEmptyModel(String name) {
		var model=Model.newInstance(name);
		model.setBlock(blockFactory.newBlock(null));
		return model;
	}

	public void store(Model model) throws IOException {
		model.save(Path.of(modelPath), model.getName());
	}
	
	private void load(Model model) throws MalformedModelException, IOException {
		try {
			model.load(Path.of(modelPath), model.getName());	
		} catch (MalformedModelException | IOException ex) {
			model.close();
			throw ex;
		}
	}

	public ModelStore optModelPath(String modelPath) {
		this.modelPath = modelPath;
		return this;
	}

	@Override
	public void close() throws Exception {
		log.info("shutting down ModelStore and free external resources");
		modelCache.values().forEach(Model::close);
		
	}
	
}
