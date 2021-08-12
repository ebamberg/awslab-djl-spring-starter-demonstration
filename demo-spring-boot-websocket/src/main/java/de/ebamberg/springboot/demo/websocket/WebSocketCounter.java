package de.ebamberg.springboot.demo.websocket;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class WebSocketCounter {
	
	private static int counter=20;
	
	@Autowired
	private SimpMessagingTemplate simpMessagingTemplate;
	
	@Scheduled (fixedRate = 5000)
	public void countDown() {
		if (counter>=0) {
			simpMessagingTemplate.convertAndSend("/topic/countdown", counter--);
		}
		
	}
	

}
