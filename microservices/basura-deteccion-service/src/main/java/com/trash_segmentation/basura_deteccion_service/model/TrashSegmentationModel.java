package com.trash_segmentation.basura_deteccion_service.model;

import java.time.LocalDateTime;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import com.fasterxml.jackson.annotation.JsonProperty;


@Document("detections")
public class TrashSegmentationModel {
    @Id
    @JsonProperty("_id")
    private String mongoId;

    @Field("id")
    @JsonProperty("id")
    private int etiqueta_id;
    private int total;
    private String dia_semana;
    private String etiqueta;
    private int conteo;
    private LocalDateTime fecha_hora;

    public TrashSegmentationModel() {}

    public TrashSegmentationModel(String mongoId, int etiqueta_id, String etiqueta, int total, String dia_semana, int conteo, LocalDateTime fecha_hora) {
        this.mongoId = mongoId;
        this.etiqueta_id = etiqueta_id;
        this.total = total;
        this.dia_semana = dia_semana;
        this.etiqueta = etiqueta;
        this.conteo = conteo;
        this.fecha_hora = fecha_hora;
    }

    public String getMongoId(){
        return mongoId;
    }

    public void setMongoId(String mongoId){
        this.mongoId = mongoId;
    }

    public int getEtiquetaId() {
        return etiqueta_id;
    }

    public void setEtiquetaId(int etiqueta_id) {
        this.etiqueta_id = etiqueta_id;
    }

    public int getTotal() {
        return total;
    }

    public void setTotal(int total) {
        this.total = total;
    }

    public String getDiaSemana() {
        return dia_semana;
    }

    public void setDiaSemana(String dia_semana) {
        this.dia_semana = dia_semana;
    }

    public String getEtiqueta() {
        return etiqueta;
    }

    public void setEtiqueta(String etiqueta) {
        this.etiqueta = etiqueta;
    }

    public int getConteo() {
        return conteo;
    }

    public void setConteo(int conteo) {
        this.conteo = conteo;
    }

    public LocalDateTime getFecha_Hora() {
        return fecha_hora;
    }

    public void setFecha_Hora(LocalDateTime fecha_hora) {
        this.fecha_hora = fecha_hora;
    }
}