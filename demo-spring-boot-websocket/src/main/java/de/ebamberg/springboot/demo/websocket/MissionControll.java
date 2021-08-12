package de.ebamberg.springboot.demo.websocket;

import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.simp.annotation.SendToUser;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class MissionControll {

	@MessageMapping ("/abortMission")
	@SendToUser ("/queue/missionStatusResponse")
	public String abortMission() {
		return "mission aborted";
	}
	
}
