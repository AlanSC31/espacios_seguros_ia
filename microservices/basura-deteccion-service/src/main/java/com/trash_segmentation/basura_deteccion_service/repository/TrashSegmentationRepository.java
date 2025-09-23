package com.trash_segmentation.basura_deteccion_service.repository;

import java.util.List;

import org.springframework.data.mongodb.repository.Aggregation;
import org.springframework.data.mongodb.repository.MongoRepository;

import com.trash_segmentation.basura_deteccion_service.dto.ConteoByDate;
import com.trash_segmentation.basura_deteccion_service.dto.ConteoByDiaPerEtiqueta;
import com.trash_segmentation.basura_deteccion_service.dto.ConteoByDiaSemana;
import com.trash_segmentation.basura_deteccion_service.dto.ProportionByEtiqueta;
import com.trash_segmentation.basura_deteccion_service.dto.TotalByEtiqueta;
import com.trash_segmentation.basura_deteccion_service.dto.TotalConteo;
import com.trash_segmentation.basura_deteccion_service.model.TrashSegmentationModel;

public interface TrashSegmentationRepository extends MongoRepository<TrashSegmentationModel, String> {

    /** Total de detecciones por cada etiqueta */
    @Aggregation(pipeline = {
        "{ $group: { _id: { etiqueta: '$etiqueta'}, total: { $sum: '$conteo' } } }",
        "{ $project: { etiqueta: '$_id.etiqueta', total: 1, _id: 0 } }"
    })
    List<TotalByEtiqueta> findTotalByEtiqueta();

    /** Proporción de cada etiqueta respecto al total de detecciones */
    @Aggregation(pipeline = {
    // Sumar por etiqueta y calcular total general
    "{ $group: { _id: '$etiqueta', conteo: { $sum: '$conteo' } } }",
    "{ $group: { _id: null, totalGeneral: { $sum: '$conteo' }, datos: { $push: { etiqueta: '$_id', conteo: '$conteo' } } } }",
    "{ $unwind: '$datos' }",
    "{ $project: { etiqueta: '$datos.etiqueta', proporcion: { $divide: [ '$datos.conteo', '$totalGeneral' ] }, _id: 0 } }"
    })
    List<ProportionByEtiqueta> findProportionByTotalEtiqueta();

    /** Conteo de detecciones agrupado por fecha (YYYY-MM-DD) */
    @Aggregation(pipeline = {
        "{ $addFields: { fecha: { $dateToString: { format: '%Y-%m-%d', date: '$fecha_hora' } } } }",
        "{ $group: { _id: '$fecha', conteo: { $sum: '$conteo' } } }",
        "{ $project: { fecha: '$_id', conteo: 1, _id: 0 } }",
        "{ $sort: { fecha: 1 } }"
    })
    List<ConteoByDate> findConteosByDate();

        /** Conteo total agrupado por día de la semana */
    @Aggregation(pipeline = {
        "{ $group: { _id: '$dia_semana', conteo: { $sum: '$conteo' } } }",
        "{ $project: { dia: '$_id', conteo: 1, _id: 0 } }",
        "{ $sort: { dia: 1 } }"
    })
    List<ConteoByDiaSemana> conteosPorDiaSemana();

    /** Conteo total solo de lunes a viernes (entre semana) */
    @Aggregation(pipeline = {
        "{ $match: { dia_semana: { $in: ['lunes', 'martes', 'miércoles', 'jueves', 'viernes'] } } }",
        "{ $group: { _id: null, conteo: { $sum: '$conteo' } } }",
        "{ $project: { _id: 0, conteo: 1 } }"
    })
    TotalConteo conteoEntreSemana();

    /** Conteo total de sábado y domingo (fin de semana) */
    @Aggregation(pipeline = {
        "{ $match: { dia_semana: { $in: ['sábado', 'domingo'] } } }",
        "{ $group: { _id: null, conteo: { $sum: '$conteo' } } }",
        "{ $project: { _id: 0, conteo: 1 } }"
    })
    TotalConteo conteoFinDeSemana();

    /** Conteo tipo de desecho por dia */
    @Aggregation(pipeline = {
        "{ $group: { _id: { dia: '$dia_semana', etiqueta: '$etiqueta' }, conteo: { $sum: '$conteo' } } }",
        "{ $project: { dia: '$_id.dia', etiqueta: '$_id.etiqueta', conteo: 1, _id: 0 } }",
        "{ $sort: { dia: 1 } }"
    })
    List<ConteoByDiaPerEtiqueta> conteoBasuraPorDia();
}