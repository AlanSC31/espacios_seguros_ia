package com.personas_detectadas.repository;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;
import com.personas_detectadas.model.Deteccion;
import java.util.List;

@Repository
public interface DeteccionRepository extends CrudRepository<Deteccion, Long> {

    @Query(value = "SELECT DAYNAME(timestamp_detectado) AS dia, COUNT(*) AS conteo " +
                   "FROM personas_detectadas GROUP BY DAYNAME(timestamp_detectado)", nativeQuery = true)
    List<Object[]> contarPorDiaDeLaSemana();
}
