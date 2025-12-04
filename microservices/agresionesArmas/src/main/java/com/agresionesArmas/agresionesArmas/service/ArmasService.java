package com.agresionesArmas.agresionesArmas.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import com.agresionesArmas.agresionesArmas.model.AlertaArma;
import com.agresionesArmas.agresionesArmas.repository.AlertaArmaRepository;

@SpringBootApplication
public class ArmasService {
    @Autowired
    private AlertaArmaRepository alertaArmaRepository;
    
    public List<AlertaArma> getAllAlertaArma() {
        return alertaArmaRepository.findAll();
    }

}
