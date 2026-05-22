# SteamAgentRecommender

> Sistema inteligente de recomendación de videojuegos RPG dentro de la plataforma Steam

---

## Descripción

**SteamAgentRecommender** es un agente inteligente diseñado para recomendar videojuegos de tipo **RPG** dentro de la plataforma **Steam**, una de las mayores tiendas de distribución digital de videojuegos del mundo.

---

## Enfoque del Proyecto

A diferencia de sistemas tradicionales, Steam utiliza un modelo de clasificación **híbrido**, lo que hace más complejo el proceso de recomendación:

### Sistema de Clasificación
- **Categorías oficiales** (géneros principales)
- **Tags descriptivos** (mecánicas, ambientación, estilo)
- **Subgéneros emergentes** (combinaciones de tags)

### Origen de las etiquetas
- **Desarrolladores** → asignación inicial  
- **Comunidad** → votación y ajuste dinámico  

---

## Problema a Resolver

Desarrollar un agente inteligente capaz de:
- Interpretar tags y relaciones entre juegos  
- Generar recomendaciones personalizadas de alta calidad  

---

## Tecnologías

- Python
- API-REST
- Sistemas de recomendación 
- Procesamiento de datos  

---
## Instrucciones

Levantar contenedores:

```bash
docker compose up -d --build
```

Ver logs en vivo:

```bash
docker compose logs -f
```

Detener stack:

```bash
docker compose down
```

```
Correr API
uvicorn src.api.app:app --reload
```

---

## Uso del Proyecto

Este repositorio **no cuenta con una licencia**.

Por lo tanto:
- No está permitido reutilizar, modificar o distribuir el código sin autorización explícita  
- Su uso es únicamente con fines académicos o de referencia  
