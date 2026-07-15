import redis
import json
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import settings

logger = logging.getLogger(__name__)


class RedisSemanticCache:

    def __init__(self, embeddings: Optional[HuggingFaceEmbeddings] = None, redis_url: Optional[str] = None):
        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        url = redis_url or settings.REDIS_URL
        self.threshold = settings.REDIS_SEMANTIC_CACHE_THRESHOLD
        
        try:
            self.redis_client = redis.Redis.from_url(url, decode_responses=True)
            self.redis_client.ping()
            self.is_connected = True
        except Exception as e:
            logger.error("Failed to connect to Redis server at %s: %s", url, e)
            self.is_connected = False
            self.redis_client = None

        self._cached_embeddings: List[np.ndarray] = []
        self._cached_keys: List[str] = []
        self._initialized = False


    def _initialize_local_cache(self):
        if not self.is_connected or not self.redis_client:
            return

        try:
            keys = self.redis_client.keys("semantic_cache:*")
            embeddings = []
            valid_keys = []
            
            for key in keys:
                emb_str = self.redis_client.hget(key, "embedding")
                if emb_str:
                    emb = json.loads(emb_str)
                    embeddings.append(np.array(emb, dtype=np.float32))
                    valid_keys.append(key)
            
            self._cached_embeddings = embeddings
            self._cached_keys = valid_keys
            self._initialized = True
            logger.info("Loaded %d semantic cache vectors from Redis.", len(embeddings))
        except Exception as e:
            logger.error("Failed to initialize semantic cache vectors: %s", e)


    async def get(self, query: str) -> Optional[Tuple[str, List[Dict[str, Any]]]]:

        if not self.is_connected or not self.redis_client:
            return None

        try:
            if not self._initialized:
                self._initialize_local_cache()

            if not self._cached_embeddings:
                return None

            query_emb = await self.embeddings.aembed_query(query)
            query_vector = np.array(query_emb, dtype=np.float32)

            similarities = []
            for cache_vector in self._cached_embeddings:
                dot_prod = np.dot(query_vector, cache_vector)
                norm_query = np.linalg.norm(query_vector)
                norm_cache = np.linalg.norm(cache_vector)
                if norm_query > 0 and norm_cache > 0:
                    similarity = dot_prod / (norm_query * norm_cache)
                else:
                    similarity = 0.0
                similarities.append(similarity)

            if not similarities:
                return None

            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]

            logger.info("Semantic Cache search: max similarity = %.4f for query='%s'", max_similarity, query[:40])

            if max_similarity >= self.threshold:
                key = self._cached_keys[max_idx]
                data = self.redis_client.hgetall(key)
                if data and "answer" in data:
                    answer = data["answer"]
                    sources = json.loads(data.get("sources", "[]"))
                    logger.info("Semantic cache HIT (similarity %.4f) on key: %s", max_similarity, key)
                    return answer, sources
            
            return None
        except Exception as e:
            logger.error("Error searching semantic cache: %s", e)
            return None


    async def set(self, query: str, answer: str, sources: List[Dict[str, Any]]) -> None:
        if not self.is_connected or not self.redis_client:
            return

        try:
            import uuid
            cache_id = str(uuid.uuid4())
            key = f"semantic_cache:{cache_id}"

            query_emb = await self.embeddings.aembed_query(query)

            self.redis_client.hset(key, mapping={
                "query": query,
                "answer": answer,
                "sources": json.dumps(sources),
                "embedding": json.dumps(query_emb)
            })
            
            self.redis_client.expire(key, 7 * 24 * 60 * 60)

            self._cached_embeddings.append(np.array(query_emb, dtype=np.float32))
            self._cached_keys.append(key)
            logger.info("Cached query semantically in Redis under key: %s", key)
        except Exception as e:
            logger.error("Failed to write to semantic cache: %s", e)


    def clear(self) -> None:
        if not self.is_connected or not self.redis_client:
            return

        try:
            keys = self.redis_client.keys("semantic_cache:*")
            if keys:
                self.redis_client.delete(*keys)
            self._cached_embeddings = []
            self._cached_keys = []
            self._initialized = True
            logger.info("Semantic cache cleared.")
        except Exception as e:
            logger.error("Failed to clear semantic cache: %s", e)
