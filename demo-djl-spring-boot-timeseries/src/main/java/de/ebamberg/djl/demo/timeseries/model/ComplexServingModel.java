package de.ebamberg.djl.demo.timeseries.model;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import ai.djl.util.PairList;
import de.ebamberg.djl.demo.timeseries.utils.MinMaxScaler;

public class ComplexServingModel implements Model {

	private Model wrappedModel;
	private Map<String,Object> components;
	
	
    protected ComplexServingModel(Model wrappedModel) {
		super();
		this.wrappedModel = wrappedModel;
		this.components=new ConcurrentHashMap<String, Object>();
	}

	/**
     * Creates an empty model instance.
     *
     * @param name the model name
     * @return a new Model instance
     */
    public static ComplexServingModel newInstance(String name) {
    	return new ComplexServingModel(Model.newInstance(name, (Device) null));
    }

    
    public void addComponent(String name, Object component) {
    	components.put(name, component);
    }
    
    public Optional<Object> getComponent(String name) {
    	if (components.containsKey(name)) {
    		return Optional.of(components.get(name));
    	} else {
    		return Optional.empty();
    	}
    	
    }
    
    @SuppressWarnings("unchecked")
	public <T> Optional<T> getComponent(Class<T> type) {
    	return components.values().stream().filter( type::isInstance).map( c->(T)c ).findFirst();
    }
    
    
	public Model getWrappedModel() {
		return wrappedModel;
	}

	public void load(Path modelPath) throws IOException, MalformedModelException {
		wrappedModel.load(modelPath);

	}

	public void load(Path modelPath, String modelName) throws IOException, MalformedModelException {
		wrappedModel.load(modelPath, modelName);
		int version=1;
        String fileName = String.format(Locale.ENGLISH, "%s-v%04d.model", modelName, version);
        Path modelFile = modelPath.resolve(fileName);
        try (DataInputStream dis = new DataInputStream(Files.newInputStream(modelFile))) {
            byte[] buf = new byte[6];
            dis.readFully(buf);
            if (!"MODEL@".equals(new String(buf, StandardCharsets.US_ASCII))) {
                throw new MalformedModelException("file is not a saved model");
            }
            int readversion = dis.readInt();
            String componentName = dis.readUTF();
            if ("scaler".equals(componentName)) {
            	MinMaxScaler scaler=new MinMaxScaler();
            //	NDArray min=dis.read(NDArray.d)
            }
        }
	}

	public void load(Path modelPath, String prefix, Map<String, ?> options)
			throws IOException, MalformedModelException {
		wrappedModel.load(modelPath, prefix, options);
	}

	public void save(Path modelPath, String newModelName) throws IOException {
		String modelName= newModelName==null ? wrappedModel.getName() : newModelName;
		int version=1;
		
		wrappedModel.save(modelPath, newModelName);
		
        String fileName = String.format(Locale.ENGLISH, "%s-v%04d.model", modelName, version);
        Path modelFile = modelPath.resolve(fileName);

        try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(modelFile))) {	
    		dos.writeBytes("MODEL@");
            dos.writeInt(version);
			getComponent(MinMaxScaler.class).ifPresent( scaler -> {
	            try {
		            dos.writeUTF("scaler");
		            dos.write(scaler.getMin().encode());
		            dos.write(scaler.getMax().encode());		      
				} catch (IOException e) {
					throw new RuntimeException(e);
				}

			});
        }
	}

	public Path getModelPath() {
		return wrappedModel.getModelPath();
	}

	public Block getBlock() {
		return wrappedModel.getBlock();
	}

	public void setBlock(Block block) {
		wrappedModel.setBlock(block);
	}

	public String getName() {
		return wrappedModel.getName();
	}

	public String getProperty(String key) {
		return wrappedModel.getProperty(key);
	}

	public void setProperty(String key, String value) {
		wrappedModel.setProperty(key, value);
	}

	public NDManager getNDManager() {
		return wrappedModel.getNDManager();
	}

	public Trainer newTrainer(TrainingConfig trainingConfig) {
		return wrappedModel.newTrainer(trainingConfig);
	}

	public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
		return wrappedModel.newPredictor(translator);
	}

	public PairList<String, Shape> describeInput() {
		return wrappedModel.describeInput();
	}

	public PairList<String, Shape> describeOutput() {
		return wrappedModel.describeOutput();
	}

	public String[] getArtifactNames() {
		return wrappedModel.getArtifactNames();
	}

	public <T> T getArtifact(String name, Function<InputStream, T> function) throws IOException {
		return wrappedModel.getArtifact(name, function);
	}

	public URL getArtifact(String name) throws IOException {
		return wrappedModel.getArtifact(name);
	}

	public InputStream getArtifactAsStream(String name) throws IOException {
		return wrappedModel.getArtifactAsStream(name);
	}

	public void setDataType(DataType dataType) {
		wrappedModel.setDataType(dataType);
	}

	public DataType getDataType() {
		return wrappedModel.getDataType();
	}

	public void cast(DataType dataType) {
		wrappedModel.cast(dataType);
	}

	public void quantize() {
		wrappedModel.quantize();
	}

	public void close() {
		wrappedModel.close();
	}


	
}
