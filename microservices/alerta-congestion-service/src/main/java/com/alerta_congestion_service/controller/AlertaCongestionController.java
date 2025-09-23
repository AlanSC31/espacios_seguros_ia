package com.alerta_congestion_service.controller;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.alerta_congestion_service.model.AlertasCongestion;
import com.alerta_congestion_service.service.AlertaService;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/alertas")
public class AlertaCongestionController {
   @Autowired
   private AlertaService alertaService; 

@GetMapping("/get")
public ResponseEntity<List<AlertasCongestion>> obtenerAlertasPorFecha(
    @RequestParam("desde") @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime desde,
    @RequestParam("hasta") @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime hasta
) {
    return ResponseEntity.ok(alertaService.obtenerPorFechas(desde, hasta));
}
}
