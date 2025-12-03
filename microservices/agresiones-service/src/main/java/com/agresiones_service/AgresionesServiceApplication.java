package com.agresiones_service;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class AgresionesServiceApplication {

    public static void main(String[] args) {
        System.out.println("⏳ Iniciando agresiones-service...");
        SpringApplication.run(AgresionesServiceApplication.class, args);
        System.out.println("✅ agresiones-service iniciado correctamente.");
    }

}
