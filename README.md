# data-processor-lambda

**Función**: Procesamiento de datos scrapeados

- **Trigger**: Eventos S3 cuando se suben archivos JSON
- **Responsabilidades**: 
    - Procesa archivos raw JSON desde S3
    - Limpia y normaliza datos de cursos
    - Genera embeddings ML con sentence-transformers
    - Clasifica automáticamente cursos por tema/nivel
    - Detecta duplicados por similaridad semántica
    - Inserta datos limpios en MongoDB
    - **No expone endpoints** (función interna)
