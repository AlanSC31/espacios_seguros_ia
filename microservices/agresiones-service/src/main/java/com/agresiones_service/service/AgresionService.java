package com.agresiones_service.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.agresiones_service.model.Agresion;
import com.agresiones_service.repository.AgresionRepository;

@Service
public class AgresionService {

    @Autowired
    private AgresionRepository repository;

    public Agresion save(Agresion agresion) {
        return repository.save(agresion);
    }

    public List<Agresion> getAll() {
        return repository.findAll();
    }

    public Agresion getById(Long id) {
        return repository.findById(id).orElse(null);
    }

    public long count() {
        return repository.count();
    }
}
