package com.agresionesArmas.agresionesArmas.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.agresionesArmas.agresionesArmas.model.AlertaArma;

@Repository
public interface AlertaArmaRepository extends JpaRepository<AlertaArma, Long> {
    
}
