package com.agresiones_service.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "agresiones_detectadas")
public class Agresion {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private LocalDateTime timestamp;

    @Column(name = "video_nombre")
    private String videoNombre;

    @Column(name = "frame_num")
    private Integer frameNum;

    @Column(name = "cant_personas")
    private Integer cantPersonas;

    @Lob
    @Column(name = "id_involucrados")
    private String idInvolucrados;

    @Lob
    @Column(name = "tipo_agresion")
    private String tipoAgresion;

    @Column(name = "distancia_entre")
    private Double distanciaEntre;

    @Lob
    @Column(name = "velocidad_manos")
    private String velocidadManos;

    @Lob
    @Column(name = "velocidad_torsos")
    private String velocidadTorsos;

    // === Getters y Setters ===

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }

    public String getVideoNombre() {
        return videoNombre;
    }

    public void setVideoNombre(String videoNombre) {
        this.videoNombre = videoNombre;
    }

    public int getFrameNum() {
        return frameNum;
    }

    public void setFrameNum(int frameNum) {
        this.frameNum = frameNum;
    }

    public int getCantPersonas() {
        return cantPersonas;
    }

    public void setCantPersonas(int cantPersonas) {
        this.cantPersonas = cantPersonas;
    }

    public String getIdInvolucrados() {
        return idInvolucrados;
    }

    public void setIdInvolucrados(String idInvolucrados) {
        this.idInvolucrados = idInvolucrados;
    }

    public String getTipoAgresion() {
        return tipoAgresion;
    }

    public void setTipoAgresion(String tipoAgresion) {
        this.tipoAgresion = tipoAgresion;
    }

    public double getDistanciaEntre() {
        return distanciaEntre;
    }

    public void setDistanciaEntre(double distanciaEntre) {
        this.distanciaEntre = distanciaEntre;
    }

    public String getVelocidadManos() {
        return velocidadManos;
    }

    public void setVelocidadManos(String velocidadManos) {
        this.velocidadManos = velocidadManos;
    }

    public String getVelocidadTorsos() {
        return velocidadTorsos;
    }

    public void setVelocidadTorsos(String velocidadTorsos) {
        this.velocidadTorsos = velocidadTorsos;
    }
}
