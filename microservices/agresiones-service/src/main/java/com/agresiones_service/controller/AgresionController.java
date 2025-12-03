package com.agresiones_service.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.agresiones_service.model.Agresion;
import com.agresiones_service.service.AgresionService;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/agresiones")
public class AgresionController {

    private final AgresionService service;

    public AgresionController(AgresionService service) {
        this.service = service;
    }

    @PostMapping
    public ResponseEntity<Agresion> createAggression(@RequestBody Agresion agresion) {
        return ResponseEntity.ok(service.save(agresion));
    }

    @GetMapping
    public ResponseEntity<List<Agresion>> getAllAggressions() {
        return ResponseEntity.ok(service.getAll());
    }

    @GetMapping("/{id}")
    public ResponseEntity<Agresion> getAggressionById(@PathVariable Long id) {
        Agresion agresion = service.getById(id);
        if (agresion != null) {
            return ResponseEntity.ok(agresion);
        }
        return ResponseEntity.notFound().build();
    }

    @GetMapping("/count")
    public ResponseEntity<Long> count() {
        return ResponseEntity.ok(service.count());
    }

    @GetMapping("/ping")
    public String ping() {
        return "pong";
    }
}
