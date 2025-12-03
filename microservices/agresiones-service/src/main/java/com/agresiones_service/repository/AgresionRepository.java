package com.agresiones_service.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.agresiones_service.model.Agresion;

@Repository
public interface AgresionRepository extends JpaRepository<Agresion, Long> {
}
