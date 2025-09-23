package com.trash_segmentation.basura_deteccion_service;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerMapping;

@SpringBootApplication(exclude = { SecurityAutoConfiguration.class })
public class TrashSegmentationApplication {

	public static void main(String[] args) {
		SpringApplication.run(TrashSegmentationApplication.class, args);
	}

	// Este bean imprimirá todas las rutas registradas al iniciar
    @Bean
    public CommandLineRunner printMappings(RequestMappingHandlerMapping mapping) {
        return args -> mapping.getHandlerMethods()
                              .forEach((key, value) -> System.out.println(key));
    }

}
