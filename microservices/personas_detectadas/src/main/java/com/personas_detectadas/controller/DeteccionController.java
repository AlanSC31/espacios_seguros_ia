package com.personas_detectadas.controller;

import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.personas_detectadas.service.DeteccionService;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/deteccion")
public class DeteccionController {

    @Autowired
    private DeteccionService deteccionService;

    @GetMapping("/por-dia-semana")
    public Map<String, Long> obtenerConteoPorDiaDeLaSemana() {
        return deteccionService.obtenerConteoPorDiaDeLaSemana();
    }
}
