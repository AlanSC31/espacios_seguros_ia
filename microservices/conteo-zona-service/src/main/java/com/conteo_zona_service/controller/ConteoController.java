package com.conteo_zona_service.controller;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.conteo_zona_service.model.ConteoZona;
import com.conteo_zona_service.service.ConteoService;

import java.time.LocalDateTime;
import java.util.List;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/conteo")
public class ConteoController {

    @Autowired
    private ConteoService service;

    // Por timestamp exacto
    @GetMapping("/por-fecha")
    public ResponseEntity<List<ConteoZona>> obtenerPorFecha(
            @RequestParam("timestamp") 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime timestamp) {
        List<ConteoZona> lista = service.obtenerPorTimestamp(timestamp);
        return ResponseEntity.ok(lista);
    }

    // Por rango de fechas (opcional)
    @GetMapping("/por-rango")
    public ResponseEntity<List<ConteoZona>> obtenerPorRango(
            @RequestParam("desde") 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime desde,
            @RequestParam("hasta") 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime hasta) {
        List<ConteoZona> lista = service.obtenerPorRango(desde, hasta);
        return ResponseEntity.ok(lista);
    }
}
