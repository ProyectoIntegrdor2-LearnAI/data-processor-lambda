# data_processor_lambda.py
import os
import json
import time
import logging
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
import boto3
from botocore.exceptions import ClientError
from pymongo import MongoClient, UpdateOne, ASCENDING, errors as mongo_errors

# =========================
# Configuración y Constantes
# =========================

# Configuración de logging estructurado para CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuración centralizada del procesador"""
    # MongoDB Atlas
    atlas_uri: str = os.getenv("ATLAS_URI", "")
    database_name: str = os.getenv("DATABASE_NAME", "learnia_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "courses")
    
    # Atlas Search Index (CRÍTICO: debe coincidir con el índice real)
    atlas_search_index: str = os.getenv("ATLAS_SEARCH_INDEX", "default")
    num_candidates_factor: int = int(os.getenv("NUM_CANDIDATES_FACTOR", "30"))
    semantic_dedup_enabled: bool = os.getenv("SEMANTIC_DEDUP_ENABLED", "true").lower() == "true"
    semantic_dedup_min_docs: int = int(os.getenv("SEMANTIC_DEDUP_MIN_DOCS", "200"))
    
    # Embeddings Provider
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "bedrock")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))

    # AWS
    s3_bucket: str = os.getenv("S3_BUCKET", "learnia-scraping-data")
    
    # Procesamiento
    batch_size: int = int(os.getenv("BATCH_SIZE", "50"))
    dedup_key: str = os.getenv("DEDUP_KEY", "url")
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    semantic_dedup_threshold: float = float(os.getenv("SEMANTIC_DEDUP_THRESHOLD", "0.97"))
    
    # Timeouts MongoDB
    connect_timeout_ms: int = int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "10000"))
    server_selection_timeout_ms: int = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "10000"))
    
    # Categorización
    categories_mapping: Dict[str, List[str]] = field(default_factory=lambda: {
        "programming": ["programming", "development", "coding", "software", "python", "javascript", "java", "react"],
        "data_science": ["data", "analytics", "machine learning", "ai", "statistics", "data science", "sql"],
        "design": ["design", "ui", "ux", "graphics", "visual", "photoshop", "figma"],
        "business": ["business", "marketing", "entrepreneurship", "management", "leadership"],
        "technology": ["technology", "cloud", "devops", "cybersecurity", "aws", "docker"],
        "general": []
    })

class ProcessingMetrics:
    """Métricas de procesamiento para CloudWatch"""
    def __init__(self):
        self.processed_count = 0
        self.duplicates_count = 0
        self.semantic_duplicates = 0
        self.errors_count = 0
        self.embedding_time_ms = 0.0
        self.bulk_time_ms = 0.0
        self.processing_start = time.perf_counter()
        
    def add_embedding_time(self, time_ms: float):
        self.embedding_time_ms += time_ms
        
    def add_bulk_time(self, time_ms: float):
        self.bulk_time_ms += time_ms
        
    def get_summary(self) -> Dict[str, Any]:
        total_time = (time.perf_counter() - self.processing_start) * 1000
        processed = max(self.processed_count, 1)
        
        return {
            "processed_documents": self.processed_count,
            "duplicates_found": self.duplicates_count,
            "semantic_duplicates": self.semantic_duplicates,
            "errors_encountered": self.errors_count,
            "avg_embedding_time_ms": self.embedding_time_ms / processed,
            "avg_bulk_time_ms": self.bulk_time_ms / max(1, self.processed_count // 50),
            "total_processing_time_ms": total_time,
            "documents_per_second": self.processed_count / max(total_time/1000, 0.001)
        }

# =========================
# Utilidades y Helpers
# =========================

def normalize_vector(vec: List[float]) -> List[float]:
    """Normaliza un vector para búsqueda por cosine similarity"""
    try:
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            logger.warning("Vector con norma cero encontrado")
            return arr.tolist()
        return (arr / norm).tolist()
    except Exception as e:
        logger.error(f"Error normalizando vector: {e}")
        return vec

def clean_text(text: str) -> str:
    """Limpia y normaliza texto para procesamiento"""
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text)  # Múltiples espacios -> uno
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)  # Solo alfanuméricos y puntuación básica
    return text.strip()

def generate_content_hash(title: str, description: str) -> str:
    """Genera hash único basado en contenido para deduplicación"""
    content = f"{title.lower().strip()}|{description.lower().strip()}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

def to_float(x, default=0.0):
    """Conversión segura a float"""
    try:
        if x in (None, "", "null", "N/A"):
            return default
        return float(x)
    except (ValueError, TypeError):
        return default

def to_int(x, default=0):
    """Conversión segura a int"""
    try:
        if x in (None, "", "null", "N/A"):
            return default
        return int(float(x))
    except (ValueError, TypeError):
        return default

def categorize_course(title: str, description: str, config: ProcessingConfig) -> str:
    """Categoriza automáticamente un curso basado en contenido"""
    text = f"{title} {description}".lower()
    
    for category, keywords in config.categories_mapping.items():
        if category == "general":
            continue
        if any(keyword in text for keyword in keywords):
            return category
    
    return "general"

def extract_course_level(description: str) -> str:
    """Extrae el nivel del curso de la descripción"""
    desc_lower = description.lower()
    
    if any(word in desc_lower for word in ["beginner", "básico", "introducción", "intro", "principiante"]):
        return "beginner"
    elif any(word in desc_lower for word in ["advanced", "avanzado", "expert", "experto", "master"]):
        return "advanced"
    elif any(word in desc_lower for word in ["intermediate", "intermedio", "medio"]):
        return "intermediate"
    else:
        return "intermediate"

# =========================
# Cache LRU en memoria (reemplazo REAL de Redis)
# =========================

from collections import OrderedDict

class EmbeddingCache:
    """Cache LRU funcional en memoria para embeddings"""
    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self.store = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[List[float]]:
        """Obtiene embedding desde cache LRU"""
        value = self.store.get(key)
        if value is not None:
            # Mover al final (más recientemente usado)
            self.store.move_to_end(key, last=True)
            self.hits += 1
            return value
        else:
            self.misses += 1
            return None

    def set(self, key: str, value: List[float]) -> None:
        """Guarda embedding en cache LRU"""
        if key in self.store:
            # Ya existe, mover al final
            self.store.move_to_end(key, last=True)
        self.store[key] = value
        
        # Eliminar el más antiguo si excede capacidad
        if len(self.store) > self.max_size:
            self.store.popitem(last=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.store),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2)
        }

# =========================
# Servicios
# =========================

class EmbeddingService:
    """Servicio unificado para embeddings con múltiples providers"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.provider = config.embedding_provider.lower()
        self.max_retries = config.max_retries
        self.cache = EmbeddingCache(max_size=int(os.getenv("EMBEDDING_CACHE_MAX", "5000")))
        
        # Inicializar únicamente Bedrock
        if self.provider != "bedrock":
            logger.info(f"Embedding provider '{self.provider}' no soportado aquí; forzando 'bedrock'")
            self.provider = "bedrock"
        self.bedrock_client = boto3.client('bedrock-runtime')
    def _generate_bedrock_embedding(self, text: str) -> List[float]:
        """Genera embedding usando AWS Bedrock (CORREGIDO)"""
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.config.embedding_model,
                body=json.dumps({"inputText": text}).encode("utf-8"),
                accept="application/json",
                contentType="application/json"
            )
            
            result = json.loads(response['body'].read())
            return result['embedding']
        except Exception as e:
            logger.error(f"Error con Bedrock: {e}")
            raise
    
    
    def generate_embedding(self, text: str) -> Tuple[List[float], float]:
        """Genera embedding con cache FUNCIONAL y retry"""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        
        # Verificar cache LRU REAL
        cached = self.cache.get(key)
        if cached is not None:
            return cached, 0.0
        
        # Generar embedding con retry
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                
                # Solo Bedrock
                embedding = self._generate_bedrock_embedding(text)
                
                embedding_time = (time.perf_counter() - t0) * 1000
                
                # Normalizar y validar dimensión
                embedding = normalize_vector(embedding)
                if len(embedding) != self.config.embedding_dim:
                    logger.warning(f"Embedding dim {len(embedding)} != {self.config.embedding_dim}")
                
                # Guardar en cache LRU REAL
                self.cache.set(key, embedding)
                
                return embedding, embedding_time
                
            except Exception as e:
                logger.warning(f"Error generando embedding (intento {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("Failed to generate embedding after all retries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache para monitoreo"""
        return self.cache.get_stats()

class S3Service:
    """Servicio para interactuar con S3 - CORREGIDO para usar bucket del evento"""
    
    def __init__(self, default_bucket: str):
        self.s3_client = boto3.client('s3')
        self.default_bucket = default_bucket
        
    def download_file(self, s3_key: str, local_path: str, bucket: Optional[str] = None) -> bool:
        """Descarga archivo desde S3 respetando bucket del evento"""
        bucket_to_use = bucket or self.default_bucket
        try:
            self.s3_client.download_file(bucket_to_use, s3_key, local_path)
            logger.info(f"Archivo descargado desde s3://{bucket_to_use}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error descargando s3://{bucket_to_use}/{s3_key}: {e}")
            return False
    
    def get_file_size(self, s3_key: str, bucket: Optional[str] = None) -> int:
        """Obtiene tamaño del archivo en S3 respetando bucket del evento"""
        bucket_to_use = bucket or self.default_bucket
        try:
            response = self.s3_client.head_object(Bucket=bucket_to_use, Key=s3_key)
            return response['ContentLength']
        except ClientError:
            return 0

class MongoService:
    """Servicio para operaciones con MongoDB Atlas"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config  # Guardar config para semantic dedup
        self.client = MongoClient(
            config.atlas_uri,
            connectTimeoutMS=config.connect_timeout_ms,
            serverSelectionTimeoutMS=config.server_selection_timeout_ms,
            retryWrites=True
        )
        self.db = self.client[config.database_name]
        self.collection = self.db[config.collection_name]
        self.dedup_key = config.dedup_key
        self.semantic_threshold = config.semantic_dedup_threshold
        self._ensure_indexes()
        
    def _ensure_indexes(self):
        """Crea índices necesarios para optimización"""
        try:
            # Índice único para deduplicación
            if self.dedup_key:
                self.collection.create_index(
                    [(self.dedup_key, ASCENDING)],
                    unique=True,
                    name=f"unique_{self.dedup_key}",
                    background=True
                )
            
            # Índices adicionales
            indexes_to_create = [
                ([("category", ASCENDING)], "idx_category"),
                ([("level", ASCENDING)], "idx_level"), 
                ([("rating", ASCENDING)], "idx_rating"),
                ([("platform", ASCENDING)], "idx_platform"),
                ([("processed_at", ASCENDING)], "idx_processed_at"),
                ([("content_hash", ASCENDING)], "idx_content_hash"),
            ]
            
            for index_spec, name in indexes_to_create:
                try:
                    self.collection.create_index(
                        index_spec,
                        name=name,
                        background=True
                    )
                except mongo_errors.OperationFailure:
                    pass  # Índice ya existe
                    
        except Exception as e:
            logger.warning(f"Error creando índices: {e}")
    
    def is_semantic_duplicate(self, embedding: List[float]) -> bool:
        """Verifica si existe duplicado semántico usando Atlas Vector Search - CORREGIDO"""
        try:
            # Verificar si deduplicación semántica está habilitada
            if not self.config.semantic_dedup_enabled:
                return False
            
            # Verificar umbral mínimo de documentos en colección
            doc_count = self.collection.estimated_document_count()
            if doc_count < self.config.semantic_dedup_min_docs:
                logger.debug(f"Colección tiene {doc_count} docs < {self.config.semantic_dedup_min_docs}, skipping semantic dedup")
                return False

            # Configurar búsqueda vectorial
            k = 1
            num_candidates = max(100, self.config.num_candidates_factor * k)
            
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.config.atlas_search_index,  # Configurable!
                        "path": "embedding",
                        "queryVector": embedding,
                        "numCandidates": num_candidates,
                        "limit": k
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "title": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {"$limit": k}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            if results and results[0].get("score", 0.0) >= self.semantic_threshold:
                logger.info(f"Duplicado semántico encontrado: {results[0].get('title', 'N/A')}, score: {results[0].get('score')}")
                return True
                
        except Exception as e:
            logger.warning(f"Error en búsqueda semántica: {e}")
            
        return False
    
    def bulk_upsert(self, operations: List[UpdateOne]) -> Dict[str, Union[int, float]]:
        """Ejecuta operaciones bulk con manejo de errores"""
        if not operations:
            return {"upserted": 0, "modified": 0, "matched": 0, "errors": 0}
            
        try:
            t0 = time.perf_counter()
            result = self.collection.bulk_write(operations, ordered=False)
            bulk_time = (time.perf_counter() - t0) * 1000
            
            return {
                "upserted": result.upserted_count,
                "modified": result.modified_count,
                "matched": result.matched_count,
                "errors": 0,
                "bulk_time_ms": bulk_time
            }
            
        except mongo_errors.BulkWriteError as bwe:
            bulk_time = 0  # No se puede medir tiempo en error
            details = bwe.details
            logger.warning(f"Algunos documentos fallaron en bulk write: {len(details.get('writeErrors', []))}")
            
            return {
                "upserted": details.get('upsertedCount', 0),
                "modified": details.get('modifiedCount', 0),
                "matched": details.get('matchedCount', 0),
                "errors": len(details.get('writeErrors', [])),
                "bulk_time_ms": bulk_time
            }
            
        except Exception as e:
            logger.error(f"Error en bulk_upsert: {e}")
            return {"upserted": 0, "modified": 0, "matched": 0, "errors": len(operations), "bulk_time_ms": 0}
    
    def close(self):
        """Cierra conexión a MongoDB"""
        try:
            self.client.close()
        except:
            pass

# =========================
# Singletons Globales (reutilización entre invocaciones)
# =========================

# Variables globales para reutilización en Lambda warm
_CONFIG = None
_EMBEDDING_SERVICE = None
_MONGO_SERVICE = None
_S3_SERVICE = None

def get_services():
    """Obtiene servicios singleton para reutilización entre invocaciones"""
    global _CONFIG, _EMBEDDING_SERVICE, _MONGO_SERVICE, _S3_SERVICE
    
    if _CONFIG is None:
        _CONFIG = ProcessingConfig()
        
    if _EMBEDDING_SERVICE is None:
        _EMBEDDING_SERVICE = EmbeddingService(_CONFIG)
        
    if _MONGO_SERVICE is None:
        _MONGO_SERVICE = MongoService(_CONFIG)
        
    if _S3_SERVICE is None:
        _S3_SERVICE = S3Service(_CONFIG.s3_bucket)
    
    return _CONFIG, _EMBEDDING_SERVICE, _MONGO_SERVICE, _S3_SERVICE

# =========================
# Procesador Principal
# =========================

class CourseDataProcessor:
    """Procesador principal de datos de cursos"""
    
    def __init__(self, config: ProcessingConfig, embedding_service: EmbeddingService, 
                 mongo_service: MongoService, s3_service: S3Service):
        self.config = config
        self.embedding_service = embedding_service
        self.mongo_service = mongo_service
        self.s3_service = s3_service
        self.metrics = ProcessingMetrics()
        
    def process_document(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Procesa un documento individual con limpieza y enriquecimiento"""
        try:
            # Extraer y limpiar datos básicos
            title = clean_text(doc.get("titulo", doc.get("title", "")))
            description = clean_text(doc.get("descripcion", doc.get("description", "")))
            
            if not title or not description:
                logger.warning("Documento sin título o descripción válidos")
                return None
            
            # Construir texto para embedding
            embedding_text = f"Título del curso: {title}. Descripción: {description}"
            
            # Generar embedding
            embedding, embed_time = self.embedding_service.generate_embedding(embedding_text)
            self.metrics.add_embedding_time(embed_time)
            
            # Verificar duplicado semántico
            if self.mongo_service.is_semantic_duplicate(embedding):
                self.metrics.semantic_duplicates += 1
                logger.info(f"Curso duplicado semánticamente omitido: {title}")
                return None
            
            # Enriquecer datos con categorización automática
            processed_doc = {
                "title": title,
                "description": description,
                "url": doc.get("url", doc.get("enlace", "")),
                "platform": doc.get("platform", doc.get("plataforma", "unknown")),
                "instructor": doc.get("instructor", doc.get("author", "")),
                "rating": to_float(doc.get("calificacion", doc.get("rating", 0))),
                "duration": doc.get("duracion", doc.get("duration", "")),
                "price": to_float(doc.get("precio", doc.get("price", 0))),
                "students_count": to_int(doc.get("estudiantes", doc.get("students", 0))),
                "language": doc.get("idioma", doc.get("language", "es")),
                
                # Campos enriquecidos
                "category": categorize_course(title, description, self.config),
                "level": extract_course_level(description),
                "content_hash": generate_content_hash(title, description),
                "embedding": embedding,
                
                # Metadatos de procesamiento
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "embedding_model": self.config.embedding_model,
                "embedding_provider": self.config.embedding_provider,
                "embedding_dim": len(embedding),
                "processing_version": "2.1"
            }
            
            # Preservar campos adicionales del documento original
            for key, value in doc.items():
                if key not in processed_doc and not key.startswith('_'):
                    processed_doc[key] = value
                    
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            self.metrics.errors_count += 1
            return None
    
    def process_batch(self, documents: List[Dict[str, Any]]) -> bool:
        """Procesa un lote de documentos"""
        operations = []
        
        for doc in documents:
            processed = self.process_document(doc)
            if not processed:
                continue
                
            # Preparar operación de upsert
            if self.config.dedup_key and processed.get(self.config.dedup_key):
                filter_query = {self.config.dedup_key: processed[self.config.dedup_key]}
            else:
                filter_query = {"content_hash": processed["content_hash"]}
                
            operations.append(
                UpdateOne(
                    filter_query,
                    {"$set": processed},
                    upsert=True
                )
            )
        
        if operations:
            result = self.mongo_service.bulk_upsert(operations)
            
            self.metrics.processed_count += int(result["upserted"]) + int(result["modified"])
            # CORREGIDO: Evitar duplicados negativos
            dups = max(int(result["matched"]) - int(result["modified"]), 0)
            self.metrics.duplicates_count += int(dups)
            self.metrics.errors_count += int(result["errors"])
            self.metrics.add_bulk_time(result.get("bulk_time_ms", 0))
            
            logger.info(json.dumps({
                "batch_processed": True,
                "upserted": result["upserted"],
                "modified": result["modified"], 
                "matched": result["matched"],
                "errors": result["errors"],
                "semantic_duplicates": self.metrics.semantic_duplicates
            }))
            
            return True
        
        return False
    
    def process_file_json(self, file_path: str) -> bool:
        """Procesa archivo JSON completo"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"Archivo muy grande ({file_size / 1024 / 1024:.1f}MB), considerar JSONL")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self._process_data_array(data)
            
        except Exception as e:
            logger.error(f"Error procesando archivo JSON {file_path}: {e}")
            return False
    
    def process_file_jsonl(self, file_path: str) -> bool:
        """Procesa archivo JSONL línea por línea (recomendado para archivos grandes)"""
        try:
            operations = []
            processed_lines = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parseando línea {line_num}: {e}")
                        continue
                    
                    processed = self.process_document(doc)
                    if not processed:
                        continue
                    
                    # Preparar operación de upsert
                    if self.config.dedup_key and processed.get(self.config.dedup_key):
                        filter_query = {self.config.dedup_key: processed[self.config.dedup_key]}
                    else:
                        filter_query = {"content_hash": processed["content_hash"]}
                        
                    operations.append(
                        UpdateOne(
                            filter_query,
                            {"$set": processed},
                            upsert=True
                        )
                    )
                    
                    # Procesar en lotes
                    if len(operations) >= self.config.batch_size:
                        result = self.mongo_service.bulk_upsert(operations)
                        self._update_metrics_from_result(result, len(operations))
                        operations.clear()
                        processed_lines += self.config.batch_size
                        
                        logger.info(f"Progreso JSONL: {processed_lines} líneas procesadas")
            
            # Procesar lote final
            if operations:
                result = self.mongo_service.bulk_upsert(operations)
                self._update_metrics_from_result(result, len(operations))
            
            return True
            
        except Exception as e:
            logger.error(f"Error procesando archivo JSONL {file_path}: {e}")
            return False
    
    def _process_data_array(self, data: List[Dict[str, Any]]) -> bool:
        """Procesa array de datos en lotes"""
        if not isinstance(data, list):
            logger.error("Los datos deben ser un array de documentos")
            return False
        
        logger.info(f"Iniciando procesamiento de {len(data)} documentos")
        
        # Procesar en lotes
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            self.process_batch(batch)
            
            # Log de progreso
            progress = min(i + self.config.batch_size, len(data))
            logger.info(f"Progreso: {progress}/{len(data)} documentos")
        
        return True
    
    def _update_metrics_from_result(self, result: Dict[str, Union[int, float]], operations_count: int):
        """Actualiza métricas desde resultado de bulk operation - CORREGIDO"""
        self.metrics.processed_count += int(result["upserted"]) + int(result["modified"])
        # CORREGIDO: Evitar duplicados negativos
        dups = max(int(result["matched"]) - int(result["modified"]), 0)
        self.metrics.duplicates_count += int(dups)
        self.metrics.errors_count += int(result["errors"])
        self.metrics.add_bulk_time(result.get("bulk_time_ms", 0))
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Retorna resumen de procesamiento con métricas para CloudWatch"""
        return self.metrics.get_summary()

# =========================
# Lambda Handler Principal
# =========================

def _process_one_s3_object(processor, s3_service, bucket, key, processed_files):
    logger.info(json.dumps({"processing_started": True, "s3_bucket": bucket, "s3_key": key}))
    file_size = s3_service.get_file_size(key, bucket=bucket)
    if file_size > 200 * 1024 * 1024:  # 200MB
        logger.error(f"Archivo demasiado grande para Lambda: {file_size / 1024 / 1024:.1f}MB")
        return
    local_file = f"/tmp/{os.path.basename(key)}"
    if not s3_service.download_file(key, local_file, bucket=bucket):
        logger.error(f"No se pudo descargar archivo: {key}")
        return
    try:
        success = False
        if key.endswith(('.jsonl', '.ndjson')):
            logger.info("Procesando como archivo JSONL")
            success = processor.process_file_jsonl(local_file)
        else:
            logger.info("Procesando como archivo JSON")
            success = processor.process_file_json(local_file)
        if success:
            processed_files.append(key)
            logger.info(json.dumps({
                "file_processed_successfully": True,
                "s3_key": key,
                "file_size_mb": file_size / 1024 / 1024
            }))
        else:
            logger.error(f"Error procesando archivo: {key}")
    finally:
        if os.path.exists(local_file):
            os.remove(local_file)
            logger.debug(f"Archivo temporal removido: {local_file}")

def lambda_handler(event, context):
    """Handler principal para AWS Lambda"""
    # Obtener servicios singleton
    config, embedding_service, mongo_service, s3_service = get_services()
    
    # Crear procesador
    processor = CourseDataProcessor(config, embedding_service, mongo_service, s3_service)
    
    processed_files = []
    
    try:
        # Procesar eventos S3 (Notification) o EventBridge (Object Created)
        if 'Records' in event:
            # S3 Notification shape
            for record in event['Records']:
                if record.get('eventSource') == 'aws:s3':
                    bucket = record['s3']['bucket']['name']
                    key = record['s3']['object']['key']
                    _process_one_s3_object(processor, s3_service, bucket, key, processed_files)
        elif 'detail' in event and 'bucket' in event['detail'] and 'object' in event['detail']:
            # EventBridge shape
            bucket = event['detail']['bucket']['name']
            key = event['detail']['object']['key']
            _process_one_s3_object(processor, s3_service, bucket, key, processed_files)
        else:
            logger.warning(f"Evento no reconocido: {json.dumps(event)}")
        
        # Obtener métricas finales
        summary = processor.get_processing_summary()
        
        # Obtener estadísticas de cache para monitoreo
        cache_stats = embedding_service.get_cache_stats()
        
        # Log estructurado para CloudWatch
        logger.info(json.dumps({
            "lambda_execution_completed": True,
            "processed_files": processed_files,
            "metrics": summary,
            "cache_stats": cache_stats
        }))
        
        # Enviar métricas custom a CloudWatch
        try:
            cloudwatch = boto3.client('cloudwatch')
            
            metrics_to_send = [
                ('ProcessedDocuments', summary['processed_documents'], 'Count'),
                ('DuplicatesFound', summary['duplicates_found'], 'Count'),
                ('SemanticDuplicates', summary['semantic_duplicates'], 'Count'),
                ('ErrorsEncountered', summary['errors_encountered'], 'Count'),
                ('AvgEmbeddingTimeMs', summary['avg_embedding_time_ms'], 'Milliseconds'),
                ('DocumentsPerSecond', summary['documents_per_second'], 'Count/Second'),
                ('CacheHitRate', cache_stats['hit_rate_percent'], 'Percent'),
                ('CacheSize', cache_stats['size'], 'Count'),
            ]
            
            metric_data = []
            for metric_name, value, unit in metrics_to_send:
                metric_data.append({
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit,
                    'Dimensions': [
                        {
                            'Name': 'Environment',
                            'Value': os.getenv('ENVIRONMENT', 'production')
                        }
                    ]
                })
            
            cloudwatch.put_metric_data(
                Namespace='LearnIA/DataProcessor',
                MetricData=metric_data
            )
            
        except Exception as e:
            logger.warning(f"No se pudieron enviar métricas a CloudWatch: {e}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing completed successfully',
                'processed_files': processed_files,
                'metrics': summary,
                'cache_stats': cache_stats
            })
        }
        
    except Exception as e:
        logger.error(json.dumps({
            "lambda_execution_error": True,
            "error": str(e),
            "error_type": type(e).__name__
        }))
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'processed_files': processed_files
            })
        }