package com.example.demo;

import java.io.ByteArrayOutputStream;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications.Classification;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;

@SpringBootApplication
@RestController
public class DemoApplication  {

	public static void main(String[] args) {
		SpringApplication.run(DemoApplication.class, args);
	}

	@javax.annotation.Resource
	private Supplier<Predictor<Image,DetectedObjects>> predictorProvider;
	
	
	@GetMapping (value="predict", produces = MediaType.IMAGE_PNG_VALUE)
	public @ResponseBody byte[] predict() throws Exception {
		System.out.println("Hello Detection");
		
		var resource=new ClassPathResource("/bus-lane-1.jpg");
		var image=ImageFactory.getInstance().fromInputStream(resource.getInputStream());
		
		try ( var predictor=predictorProvider.get()) {
			var detectedObjects=predictor.predict(image);
			detectedObjects.items().forEach( item -> System.out.println(item.getClassName()));
			image.drawBoundingBoxes(detectedObjects);
		}
		ByteArrayOutputStream outStream=new ByteArrayOutputStream();
		image.save(outStream, "png");
		return outStream.toByteArray();
	}

	
	/**
	 * curl -X POST http://127.0.0.1:8080/upload -F "image=@./src/main/resources/bus-lane-1.jpg" --output testoutput.png
 
	 * @param file
	 * @return
	 * @throws Exception
	 */
	@PostMapping (value="upload", produces = MediaType.IMAGE_PNG_VALUE)
	public @ResponseBody byte[] upload(@RequestParam("image") MultipartFile file) throws Exception {
		System.out.println("Hello Detection");
		
		var image=ImageFactory.getInstance().fromInputStream(file.getInputStream());
		
		try ( var predictor=predictorProvider.get()) {
			var detectedObjects=predictor.predict(image);
			detectedObjects.items().forEach( item -> System.out.println(item.getClassName()));
			image.drawBoundingBoxes(detectedObjects);
		}
		ByteArrayOutputStream outStream=new ByteArrayOutputStream();
		image.save(outStream, "png");
		return outStream.toByteArray();
	}
	
	/**
	 * curl -X POST http://127.0.0.1:8080/listObjectsInPicture -F "image=@./src/main/resources/bus-lane-1.jpg" 
 
	 * @param file
	 * @return
	 * @throws Exception
	 */
	@PostMapping (value="listObjectsInPicture")
	public List<String> listObjectsInPicture(@RequestParam("image") MultipartFile file) throws Exception {
		System.out.println("Hello Detection");
		
		var image=ImageFactory.getInstance().fromInputStream(file.getInputStream());
		
		try ( var predictor=predictorProvider.get()) {
			var detectedObjects=predictor.predict(image);
			return detectedObjects.items().stream().map(Classification::getClassName).collect(Collectors.toList());
			
		}

	}
	
}
