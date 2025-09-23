package com.gestion_usuarios_service.service;

import com.gestion_usuarios_service.model.Usuario;
import com.gestion_usuarios_service.repository.UsuarioRepository;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class UsuarioService {
    @Autowired
    private UsuarioRepository usuarioRep;

    @Autowired
    private BCryptPasswordEncoder passwordEncoder;

    public String register(Usuario usuario) {
        if (usuarioRep.findByNombre(usuario.getNombre()) != null) {
            throw new RuntimeException("Ya existe una cuenta con este nombre");
        }

        usuario.setPassword(passwordEncoder.encode(usuario.getPassword()));
        usuarioRep.save(usuario);

        return "Usuario registrado con exito";
    }

public Usuario login(String nombre, String password) {
    Usuario usuario = usuarioRep.findByNombre(nombre);
    if (usuario == null || !passwordEncoder.matches(password, usuario.getPassword())) {
        throw new RuntimeException("Credenciales incorrectas");
    }

    return usuario; // devuelves el objeto completo
}

    public List<Usuario> getActiveUsers() {
        return usuarioRep.findByStatusTrue();
    }

    public Boolean resetPassword(Long idUsuario, String nuevaPassword) {
        Optional<Usuario> optUsuario = usuarioRep.findById(idUsuario);
        if (optUsuario.isPresent()) {
            Usuario usuario = optUsuario.get();
            String PassEncriptada = passwordEncoder.encode(nuevaPassword);
            usuario.setPassword(PassEncriptada);
            usuarioRep.save(usuario);
            return true;
        }
        return false;
    }

    public Boolean changeRol(Long idUsuario, int nuevoRol) {
        Optional<Usuario> optUsuario = usuarioRep.findById(idUsuario);
        if (optUsuario.isPresent()) {
            Usuario usuario = optUsuario.get();
            usuario.setRol(nuevoRol);
            usuarioRep.save(usuario);
            return true;
        }
        return false;
    }

    public Boolean changeStatus(Long idUsuario, boolean nuevoStatus) {
        Optional<Usuario> optUsuario = usuarioRep.findById(idUsuario);
        if (optUsuario.isPresent()) {
            Usuario usuario = optUsuario.get();
            usuario.setStatus(nuevoStatus);
            usuarioRep.save(usuario);
            return true;
        }
        return false;
    }
}
