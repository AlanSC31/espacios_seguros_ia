package com.gestion_usuarios_service.controller;

import com.gestion_usuarios_service.model.Usuario;
import com.gestion_usuarios_service.service.UsuarioService;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/usuarios")
public class UsuarioController {
    @Autowired
    private UsuarioService usuarioService;

    @PostMapping("/registro")
    public ResponseEntity<String> register(@RequestBody Usuario usuario) {
        try {
            String msg = usuarioService.register(usuario);
            return ResponseEntity.ok(msg);
        } catch (RuntimeException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }

    @PostMapping("/login")
    public ResponseEntity<Usuario> login(@RequestBody Map<String, String> body) {
        try {
            String nombre = body.get("nombre");
            String password = body.get("password");
            Usuario usuario = usuarioService.login(nombre, password); 

            return ResponseEntity.ok(usuario); 
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build(); 
        }
    }

    @GetMapping("/show")
    public List<Usuario> listarUsuarios() {
        return usuarioService.getActiveUsers();
    }

    @PutMapping("/{id}/password")
    public ResponseEntity<String> resetPassword(@PathVariable Long id, @RequestBody String nuevaPass) {
        boolean cambiado = usuarioService.resetPassword(id, nuevaPass);
        if (cambiado) {
            return ResponseEntity.ok("Contraseña actualizada con exito");

        } else {
            return ResponseEntity.status(404).body("Usuario no encontrado");
        }
    }

    @PutMapping("/{id}/rol")
    public ResponseEntity<String> changeRol(@PathVariable Long id, @RequestBody int nuevoRol) {
        boolean cambiado = usuarioService.changeRol(id, nuevoRol);
        if (cambiado) {
            return ResponseEntity.ok("Rol actualizado con exito");
        } else {
            return ResponseEntity.status(404).body("Usuario no encontrado");
        }
    }

    @PutMapping("/{id}/status")
    public ResponseEntity<String> changeStatus(@PathVariable Long id, @RequestBody boolean nuevoStatus) {
        boolean cambiado = usuarioService.changeStatus(id, nuevoStatus);
        if (cambiado) {
            return ResponseEntity.ok("Status actualizado con exito");
        } else {
            return ResponseEntity.status(404).body("Usuario no encontrado");
        }
    }

}
