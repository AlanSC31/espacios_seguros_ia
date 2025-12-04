package com.agresionesArmas.agresionesArmas.controller;

import java.util.List; 
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.agresionesArmas.agresionesArmas.model.AlertaArma;
import com.agresionesArmas.agresionesArmas.service.ArmasService;

@RestController
@RequestMapping("/api/armas")
public class ArmasController {
    private final ArmasService armasService;

    public ArmasController(ArmasService armasService) {
        this.armasService = armasService;
    }

    @GetMapping("/alertas")
    // Cambiamos el retorno a List<AlertaArma>
    public ResponseEntity<List<AlertaArma>> getAlertaArma(){
        return ResponseEntity.ok(armasService.getAllAlertaArma());
    }
}