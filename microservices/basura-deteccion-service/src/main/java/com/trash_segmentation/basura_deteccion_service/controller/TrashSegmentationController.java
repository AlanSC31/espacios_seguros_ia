package com.trash_segmentation.basura_deteccion_service.controller;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.bson.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoDatabase;
import com.trash_segmentation.basura_deteccion_service.dto.ConteoByDate;
import com.trash_segmentation.basura_deteccion_service.dto.ConteoByDiaPerEtiqueta;
import com.trash_segmentation.basura_deteccion_service.dto.ConteoByDiaSemana;
import com.trash_segmentation.basura_deteccion_service.dto.ProportionByEtiqueta;
import com.trash_segmentation.basura_deteccion_service.dto.TotalByEtiqueta;
import com.trash_segmentation.basura_deteccion_service.dto.TotalConteo;
import com.trash_segmentation.basura_deteccion_service.model.TrashSegmentationModel;
import com.trash_segmentation.basura_deteccion_service.repository.TrashSegmentationRepository;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:5500")
public class TrashSegmentationController {

    @Autowired
    private MongoTemplate mongoTemplate;

    private final TrashSegmentationRepository repo;

    public TrashSegmentationController(TrashSegmentationRepository repo) {
        this.repo = repo;
    }

    @GetMapping
    public String liveCheck() {
        return "Ol rait";
    }

    @GetMapping("/test-count")
    public long testCount() {
        // Debe devolver >0 si realmente hay datos en "detecciones_imagenes"
        return mongoTemplate.getCollection("detections").countDocuments();
    }

    @GetMapping("/db-info")
    public Map<String, Object> dbInfo() {
        String dbName = mongoTemplate.getDb().getName();
        List<String> cols = new ArrayList<>();
        mongoTemplate.getDb().listCollectionNames().into(cols);
        return Map.of("database", dbName, "collections", cols);
    }

    @Autowired
    private MongoClient mongoClient;

    @GetMapping("/all-dbs")
    public List<String> listAllDbs() {
        return mongoClient.listDatabaseNames().into(new ArrayList<>());
    }

    @GetMapping("/ping")
    public String pingMongo() {
        try {
            MongoDatabase database = mongoClient.getDatabase("MongoDB");
            Document result = database.runCommand(new Document("ping", 1));
            return "MongoDB PING OK: " + result.toJson();
        } catch (Exception e) {
            return "MongoDB PING FAILED: " + e.getMessage();
        }
    }

    @GetMapping("/detections")
    public List<TrashSegmentationModel> getAllDetections() {
        return repo.findAll();
    }

    @GetMapping("/totales")
    public List<TotalByEtiqueta> totalesPorEtiqueta() {
        return repo.findTotalByEtiqueta();
    }

    @GetMapping("/proporciones")
    public List<ProportionByEtiqueta> proporcionesPorEtiqueta() {
        return repo.findProportionByTotalEtiqueta();
    }

    @GetMapping("/por-fecha")
    public List<ConteoByDate> conteosPorFecha() {
        return repo.findConteosByDate();
    }

    @GetMapping("/por-dia-semana")
    public List<ConteoByDiaSemana> conteosPorDiaSemana() {
        return repo.conteosPorDiaSemana();
    }

    @GetMapping("/conteo/entre-semana")
    public TotalConteo entreSemana() {
        return repo.conteoEntreSemana();
    }

    @GetMapping("/conteo/fin-semana")
    public TotalConteo finDeSemana() {
        return repo.conteoFinDeSemana();
    }

    @GetMapping("/conteo/desecho-dia")
    public List<ConteoByDiaPerEtiqueta> porDesecho() {
        return repo.conteoBasuraPorDia();
    }
}