package com.personas_detectadas.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "personas_detectadas") 
public class Deteccion {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(name = "persona_id")
    private Integer personaId;

    @Column(name = "timestamp_detectado")
    private LocalDateTime timestampDetectado;

    // getters and setters

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public Integer getPersonaId() {
        return personaId;
    }

    public void setPersonaId(Integer personaId) {
        this.personaId = personaId;
    }

    public LocalDateTime getTimestampDetectado() {
        return timestampDetectado;
    }

    public void setTimestampDetectado(LocalDateTime timestampDetectado) {
        this.timestampDetectado = timestampDetectado;
    }
}