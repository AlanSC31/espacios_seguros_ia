package com.personas_detectadas.service;

import java.util.HashMap;
import java.util.Map;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.personas_detectadas.repository.DeteccionRepository;

@Service
public class DeteccionService {

    @Autowired
    DeteccionRepository deteccionRep;

    public Map<String, Long> obtenerConteoPorDiaDeLaSemana() {
        List<Object[]> resultados = deteccionRep.contarPorDiaDeLaSemana();
        Map<String, Long> mapa = new HashMap<>();

        for (Object[] fila : resultados) {
            String dia = (String) fila[0];
            Long conteo = ((Number) fila[1]).longValue();
            mapa.put(dia, conteo);
        }

        return mapa;
    }
}
