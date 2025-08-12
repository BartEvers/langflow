from __future__ import annotations

from typing import Any
import os
import json

from langflow.logging import logger


class Retriever:
    """Local vector retriever using Chroma, with lazy imports.

    If Chroma or sentence-transformers are not installed, operations will raise a
    RuntimeError with a helpful message.
    """

    def __init__(self, path: str = "data/airelius_index", collection: str = "components") -> None:
        self._path = path
        self._collection_name = collection
        self._client = None
        self._col = None
        self._embedder = None
        self._mode: str | None = None  # "chroma" or "simple"
        self._simple_store_path = os.path.join(self._path, f"{self._collection_name}.jsonl")
        self._simple_docs: list[dict[str, Any]] = []  # [{id, text, meta, emb}]
        # Simple backend configuration: 'st' (sentence-transformers) or 'tfidf'
        self._simple_method: str = os.environ.get("AIRELIUS_SIMPLE_METHOD", "st").lower()
        self._tfidf_vect = None
        self._tfidf_matrix = None

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        # Limit parallelism to reduce memory usage
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e2:  # noqa: BLE001
            raise RuntimeError(
                f"sentence-transformers import failed: {type(e2).__name__}: {e2}. Try: pip install sentence-transformers torch"
            ) from e2
        try:
            import torch  # type: ignore

            if hasattr(torch, "set_num_threads"):
                torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
        except Exception:
            pass
        model_name = os.environ.get("AIRELIUS_EMBED_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
        self._embedder = SentenceTransformer(model_name, device="cpu")
        return self._embedder

    def _ensure_backend(self) -> None:
        if self._client is not None:
            logger.debug("[PFU-RAG] Backend already initialized, mode: %s", self._mode)
            return
        
        logger.info("[PFU-RAG] Initializing backend...")
        logger.info(f"[PFU-RAG] Path: {self._path}")
        logger.info(f"[PFU-RAG] Collection: {self._collection_name}")
        logger.info(f"[PFU-RAG] Simple store path: {self._simple_store_path}")
        
        # Allow forcing simple mode via env
        if os.environ.get("AIRELIUS_RETRIEVER_MODE", "").lower() == "simple":
            logger.info("[PFU-RAG] Forcing simple mode via environment variable")
            os.makedirs(self._path, exist_ok=True)
            self._mode = "simple"
            if os.path.isfile(self._simple_store_path):
                try:
                    logger.info(f"[PFU-RAG] Loading documents from simple store: {self._simple_store_path}")
                    with open(self._simple_store_path, "r", encoding="utf-8") as fh:
                        lines = [line.strip() for line in fh if line.strip()]
                        logger.info(f"[PFU-RAG] Found {len(lines)} non-empty lines in file")
                        self._simple_docs = [json.loads(line) for line in lines]
                        logger.info(f"[PFU-RAG] Successfully loaded {len(self._simple_docs)} documents from simple store")
                        
                        # Log some sample documents for debugging
                        if self._simple_docs:
                            logger.info(f"[PFU-RAG] Sample document keys: {list(self._simple_docs[0].keys())}")
                            logger.debug(f"[PFU-RAG] First document: {self._simple_docs[0]}")
                except Exception as e:
                    logger.error(f"[PFU-RAG] Failed to load simple store: {e}")
                    self._simple_docs = []
            else:
                logger.warning(f"[PFU-RAG] Simple store file does not exist: {self._simple_store_path}")
                self._simple_docs = []
            
            # Initialize TF-IDF vectorizer if configured and we have documents
            if self._simple_method == "tfidf" and self._simple_docs:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
                    logger.info(f"[PFU-RAG] Building TF-IDF vectorizer for {len(self._simple_docs)} documents")
                    texts = [rec.get("text", "") for rec in self._simple_docs]
                    self._tfidf_vect = TfidfVectorizer(max_features=50000)
                    self._tfidf_matrix = self._tfidf_vect.fit_transform(texts)
                    logger.info(f"[PFU-RAG] TF-IDF vectorizer built successfully, matrix shape: {self._tfidf_matrix.shape}")
                except Exception as e:
                    logger.error(f"[PFU-RAG] Failed to build TF-IDF vectorizer: {e}")
                    # Don't raise here, just log the error and continue
                    self._tfidf_vect = None
                    self._tfidf_matrix = None
            
            logger.info(f"[PFU-RAG] Retriever (simple, forced) initialized. Persisted at {self._simple_store_path}")
            return

        # Try Chroma first (no embedder yet)
        logger.info("[PFU-RAG] Attempting to initialize ChromaDB")
        try:
            from chromadb import PersistentClient  # type: ignore
            self._client = PersistentClient(path=self._path)
            self._col = self._client.get_or_create_collection(
                name=self._collection_name, metadata={"hnsw:space": "cosine"}
            )
            
            # Check if ChromaDB collection actually has data
            try:
                collection_count = self._col.count()
                logger.info(f"[PFU-RAG] ChromaDB collection count: {collection_count}")
                if collection_count > 0:
                    self._mode = "chroma"
                    logger.info(f"[PFU-RAG] Retriever (Chroma) initialized at {self._path} collection={self._collection_name} with {collection_count} items")
                    return
                else:
                    logger.warning(f"[PFU-RAG] ChromaDB collection exists but is empty (count={collection_count}). Falling back to simple mode.")
            except Exception as e:
                logger.warning(f"[PFU-RAG] ChromaDB collection count check failed ({e}). Falling back to simple mode.")
                
        except Exception as e:
            logger.warning(f"[PFU-RAG] Chroma unavailable ({e}). Falling back to simple in-process index.")

        # Fallback: simple in-process index with numpy cosine search (embedder lazy)
        logger.info("[PFU-RAG] Falling back to simple mode")
        os.makedirs(self._path, exist_ok=True)
        self._mode = "simple"
        logger.info(f"[PFU-RAG] Simple store path: {self._simple_store_path}")
        logger.info(f"[PFU-RAG] Simple store file exists: {os.path.isfile(self._simple_store_path)}")
        
        if os.path.isfile(self._simple_store_path):
            try:
                logger.info(f"[PFU-RAG] Loading existing documents from simple store")
                with open(self._simple_store_path, "r", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh if line.strip()]
                    logger.info(f"[PFU-RAG] Found {len(lines)} non-empty lines in file")
                    self._simple_docs = [json.loads(line) for line in lines]
                    logger.info(f"[PFU-RAG] Successfully loaded {len(self._simple_docs)} documents from simple store")
                    
                    # Log some sample documents for debugging
                    if self._simple_docs:
                        logger.info(f"[PFU-RAG] Sample document keys: {list(self._simple_docs[0].keys())}")
                        logger.debug(f"[PFU-RAG] First document: {self._simple_docs[0]}")
            except Exception as e:
                logger.error(f"[PFU-RAG] Failed to load simple store: {e}")
                self._simple_docs = []
        else:
            logger.warning(f"[PFU-RAG] Simple store file does not exist, starting with empty index")
            self._simple_docs = []
        
        logger.info(f"[PFU-RAG] Retriever (simple, fallback) initialized with {len(self._simple_docs)} documents")
        
        # Build TF-IDF vectorizer if configured and we have documents
        if self._simple_method == "tfidf" and self._simple_docs:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
                logger.info(f"[PFU-RAG] Building TF-IDF vectorizer for {len(self._simple_docs)} documents")
                texts = [rec.get("text", "") for rec in self._simple_docs]
                self._tfidf_vect = TfidfVectorizer(max_features=50000)
                self._tfidf_matrix = self._tfidf_vect.fit_transform(texts)
                logger.info(f"[PFU-RAG] TF-IDF vectorizer built successfully, matrix shape: {self._tfidf_matrix.shape}")
            except Exception as e:
                logger.error(f"[PFU-RAG] Failed to build TF-IDF vectorizer: {e}")
                # Don't raise here, just log the error and continue
                self._tfidf_vect = None
                self._tfidf_matrix = None

    def reset(self) -> None:
        self._ensure_backend()
        if self._mode == "chroma":
            try:
                self._client.delete_collection(self._collection_name)  # type: ignore[attr-defined]
            except Exception:
                pass
            self._col = self._client.get_or_create_collection(
                name=self._collection_name, metadata={"hnsw:space": "cosine"}
            )
        else:
            self._simple_docs = []
            self._tfidf_vect = None
            self._tfidf_matrix = None
            try:
                if os.path.isfile(self._simple_store_path):
                    os.remove(self._simple_store_path)
            except Exception:
                pass

    def upsert(self, docs: list[dict[str, Any]]) -> int:
        """Upsert documents.

        docs: [{"id": str, "text": str, "meta": dict}]
        Returns number of upserted docs.
        """
        self._ensure_backend()
        logger.info(f"[PFU-RAG] Upserting {len(docs)} documents in {self._mode} mode")
        
        if not docs:
            return 0
        
        ids = [str(d["id"]) for d in docs]
        texts = [str(d["text"]) for d in docs]
        metadatas = [d.get("meta", {}) for d in docs]

        if self._mode == "chroma":
            logger.info("[PFU-RAG] Using ChromaDB for upsert")
            embedder = self._get_embedder()
            embeddings = embedder.encode(
                texts, convert_to_numpy=True, batch_size=min(8, max(1, len(texts))), show_progress_bar=False
            ).tolist()
            self._col.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            logger.info(f"[PFU-RAG] Successfully upserted {len(docs)} documents to ChromaDB")
            return len(docs)
        else:
            # Simple backends
            logger.info(f"[PFU-RAG] Using simple mode ({self._simple_method}) for upsert")
            if self._simple_method == "tfidf":
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        f"scikit-learn not available for TF-IDF: {type(e).__name__}: {e}. Try: pip install scikit-learn"
                    ) from e
                # Append documents
                count = 0
                os.makedirs(self._path, exist_ok=True)
                logger.info(f"[PFU-RAG] Appending {len(docs)} documents to TF-IDF store")
                with open(self._simple_store_path, "a", encoding="utf-8") as fh:
                    for _id, txt, meta in zip(ids, texts, metadatas):
                        rec = {"id": _id, "text": txt, "meta": meta}
                        self._simple_docs.append(rec)
                        fh.write(json.dumps(rec) + "\n")
                        count += 1
                logger.info(f"[PFU-RAG] Appended {count} documents to TF-IDF store")
                
                # Refit vectorizer on all texts (acceptable for moderate corpora)
                logger.info(f"[PFU-RAG] Rebuilding TF-IDF vectorizer for {len(self._simple_docs)} total documents")
                texts_all = [rec.get("text", "") for rec in self._simple_docs]
                self._tfidf_vect = TfidfVectorizer(max_features=50000)
                self._tfidf_matrix = self._tfidf_vect.fit_transform(texts_all)
                logger.info(f"[PFU-RAG] TF-IDF vectorizer rebuilt successfully, matrix shape: {self._tfidf_matrix.shape}")
                return count
            else:
                # sentence-transformers simple mode
                try:
                    import numpy as np  # type: ignore
                except Exception as e_np:
                    raise RuntimeError(
                        f"numpy import failed: {type(e_np).__name__}: {e_np}. Try: pip install numpy"
                    ) from e_np
                
                logger.info("[PFU-RAG] Generating embeddings for documents")
                embedder = self._get_embedder()
                embs = embedder.encode(
                    texts, convert_to_numpy=True, batch_size=min(8, max(1, len(texts))), show_progress_bar=False
                )
                logger.info(f"[PFU-RAG] Generated embeddings with shape: {embs.shape}")
                
                count = 0
                # append and persist
                os.makedirs(self._path, exist_ok=True)
                logger.info(f"[PFU-RAG] Appending {len(docs)} documents with embeddings to simple store")
                with open(self._simple_store_path, "a", encoding="utf-8") as fh:
                    for _id, txt, meta, emb in zip(ids, texts, metadatas, embs):
                        rec = {"id": _id, "text": txt, "meta": meta, "emb": emb.tolist()}
                        self._simple_docs.append(rec)
                        fh.write(json.dumps(rec) + "\n")
                        count += 1
                logger.info(f"[PFU-RAG] Successfully appended {count} documents with embeddings to simple store")
                logger.info(f"[PFU-RAG] Total documents in memory: {len(self._simple_docs)}")
                return count

    def query(self, query_text: str, k: int = 8) -> list[dict[str, Any]]:
        self._ensure_backend()
        logger.info(f"[PFU-RAG] Query mode: {self._mode}, query: {query_text[:50]}...")
        logger.info(f"[PFU-RAG] Simple docs count: {len(self._simple_docs) if self._mode == 'simple' else 'N/A'}")
        
        if self._mode == "chroma":
            try:
                embedder = self._get_embedder()
                logger.debug(f"[PFU-RAG] Using ChromaDB mode, collection count: {self._col.count()}")
                embedding = embedder.encode([query_text], convert_to_numpy=True, batch_size=1, show_progress_bar=False).tolist()
                res = self._col.query(query_embeddings=embedding, n_results=max(1, k))
                logger.debug(f"[PFU-RAG] ChromaDB query result: {res}")
                out: list[dict[str, Any]] = []
                ids = res.get("ids", [[]])[0]
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                for _id, _doc, _meta in zip(ids, docs, metas):
                    out.append({"id": _id, "text": _doc, "meta": _meta})
                logger.debug(f"[PFU-RAG] ChromaDB returning {len(out)} results")
                return out
            except Exception as e:
                logger.error(f"[PFU-RAG] ChromaDB query failed: {e}")
                return []
        else:
            logger.info(f"[PFU-RAG] Using simple mode with {len(self._simple_docs)} documents")
            if not self._simple_docs:
                logger.error("[PFU-RAG] No simple docs available for querying - this is the problem!")
                logger.error(f"[PFU-RAG] Simple store path: {self._simple_store_path}")
                logger.error(f"[PFU-RAG] Simple store exists: {os.path.isfile(self._simple_store_path)}")
                if os.path.isfile(self._simple_store_path):
                    try:
                        with open(self._simple_store_path, "r", encoding="utf-8") as fh:
                            file_content = fh.read()
                            logger.error(f"[PFU-RAG] File size: {len(file_content)} bytes")
                            logger.error(f"[PFU-RAG] First 500 chars: {file_content[:500]}")
                    except Exception as e:
                        logger.error(f"[PFU-RAG] Failed to read file: {e}")
                return []
            
            logger.info(f"[PFU-RAG] Simple method: {self._simple_method}")
            
            if self._simple_method == "tfidf":
                try:
                    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        f"scikit-learn not available for TF-IDF: {type(e).__name__}: {e}. Try: pip install scikit-learn"
                    ) from e
                if self._tfidf_vect is None or self._tfidf_matrix is None:
                    logger.error("[PFU-RAG] TF-IDF vectorizer not available")
                    return []
                logger.info("[PFU-RAG] Using TF-IDF similarity search")
                qv = self._tfidf_vect.transform([query_text])
                sims = cosine_similarity(qv, self._tfidf_matrix).ravel()
                logger.debug(f"[PFU-RAG] TF-IDF similarities shape: {sims.shape}, max: {sims.max()}, min: {sims.min()}")
                # Get top-k indices
                top_idx = sims.argsort()[::-1][: max(1, k)]
                results = []
                for idx in top_idx:
                    rec = self._simple_docs[int(idx)]
                    results.append({"id": rec.get("id"), "text": rec.get("text"), "meta": rec.get("meta", {})})
                logger.info(f"[PFU-RAG] TF-IDF returning {len(results)} results")
                return results
            else:
                logger.info("[PFU-RAG] Using embedding-based similarity search")
                # Embedding-based search - handle both pre-computed and on-the-fly embeddings
                try:
                    import numpy as np  # type: ignore
                except Exception as e_np:
                    raise RuntimeError(
                        f"numpy import failed: {type(e_np).__name__}: {e_np}. Try: pip install numpy"
                    ) from e_np
                
                embedder = self._get_embedder()
                logger.info("[PFU-RAG] Generating query embedding")
                try:
                    q = embedder.encode([query_text], convert_to_numpy=True, batch_size=1, show_progress_bar=False)[0]
                    logger.debug(f"[PFU-RAG] Query embedding shape: {q.shape}, dtype: {q.dtype}")
                    # normalize
                    q = q / (np.linalg.norm(q) + 1e-12)
                    logger.debug(f"[PFU-RAG] Query embedding norm after normalization: {np.linalg.norm(q)}")
                except Exception as e:
                    logger.error(f"[PFU-RAG] Failed to generate query embedding: {e}")
                    return []
                
                scores: list[tuple[float, dict[str, Any]]] = []
                
                logger.info(f"[PFU-RAG] Computing similarities for {len(self._simple_docs)} documents")
                for i, rec in enumerate(self._simple_docs):
                    # Check if we have pre-computed embeddings
                    if "emb" in rec and rec.get("emb"):
                        v = np.asarray(rec.get("emb", []), dtype=float)
                        if v.size == 0:
                            logger.warning(f"[PFU-RAG] Doc {i} has empty embedding, skipping")
                            continue
                        logger.debug(f"[PFU-RAG] Using pre-computed embedding for doc {i}, shape: {v.shape}")
                    else:
                        # Generate embedding on-the-fly for this document
                        try:
                            doc_text = rec.get("text", "")
                            if not doc_text:
                                logger.warning(f"[PFU-RAG] Doc {i} has no text, skipping")
                                continue
                            logger.info(f"[PFU-RAG] Generating embedding on-the-fly for doc {i}")
                            v = embedder.encode([doc_text], convert_to_numpy=True, batch_size=1, show_progress_bar=False)[0]
                            logger.debug(f"[PFU-RAG] Generated embedding for doc {i}, shape: {v.shape}")
                        except Exception as e:
                            logger.error(f"[PFU-RAG] Failed to generate embedding for doc {i}: {e}")
                            continue
                    
                    # Normalize document embedding
                    v = v / (np.linalg.norm(v) + 1e-12)
                    sim = float(np.dot(q, v))
                    scores.append((sim, rec))
                    logger.debug(f"[PFU-RAG] Doc {i} similarity: {sim:.4f}")
                
                logger.info(f"[PFU-RAG] Computed similarities for {len(scores)} documents")
                if scores:
                    logger.debug(f"[PFU-RAG] Similarity scores range: {min(s[0] for s in scores):.4f} to {max(s[0] for s in scores):.4f}")
                
                scores.sort(key=lambda x: x[0], reverse=True)
                top = [
                    {"id": r.get("id"), "text": r.get("text"), "meta": r.get("meta", {})}
                    for _, r in scores[: max(1, k)]
                ]
                logger.info(f"[PFU-RAG] Embedding search returning {len(top)} results")
                return top

    # ---------- Introspection helpers ----------
    def count(self) -> int:
        self._ensure_backend()
        if self._mode == "chroma":
            try:
                return int(self._col.count())
            except Exception:
                got = self._col.get(limit=1, include=["ids"])  # type: ignore[arg-type]
                return len(got.get("ids", []))
        else:
            return len(self._simple_docs)

    def sample(self, n: int = 5) -> list[dict[str, Any]]:
        """Return n sample documents for debugging."""
        self._ensure_backend()
        if self._mode == "chroma":
            try:
                if self._col.count() > 0:
                    res = self._col.get(limit=n)
                    docs = res.get("documents", [])
                    metas = res.get("metadatas", [])
                    return [{"text": doc, "meta": meta} for doc, meta in zip(docs, metas)]
                else:
                    return []
            except Exception:
                return []
        else:
            return self._simple_docs[:n] if self._simple_docs else []

    def by_path(self, path: str, limit: int = 20) -> list[dict[str, Any]]:
        self._ensure_backend()
        if self._mode == "chroma":
            try:
                res = self._col.get(
                    where={"path": path},  # type: ignore[arg-type]
                    limit=max(1, limit),
                    include=["ids", "documents", "metadatas"],
                )
            except Exception:
                return []
            ids = res.get("ids", [])
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
            out: list[dict[str, Any]] = []
            for _id, _doc, _meta in zip(ids, docs, metas):
                out.append({"id": _id, "text": _doc, "meta": _meta})
            return out
        else:
            return [r for r in self._simple_docs if r.get("meta", {}).get("path") == path][: max(1, limit)]

    def debug_info(self) -> dict[str, Any]:
        """Get debug information about the retriever state."""
        self._ensure_backend()
        info = {
            "mode": self._mode,
            "path": self._path,
            "collection": self._collection_name,
            "simple_store_path": self._simple_store_path,
            "simple_method": self._simple_method,
            "simple_docs_count": len(self._simple_docs),
            "environment": {
                "AIRELIUS_RETRIEVER_MODE": os.environ.get("AIRELIUS_RETRIEVER_MODE"),
                "AIRELIUS_SIMPLE_METHOD": os.environ.get("AIRELIUS_SIMPLE_METHOD"),
                "AIRELIUS_EMBED_MODEL": os.environ.get("AIRELIUS_EMBED_MODEL"),
            }
        }
        
        if self._mode == "chroma":
            try:
                info["chroma_count"] = self._col.count() if self._col else None
            except Exception:
                info["chroma_count"] = "error"
        else:
            info["tfidf_available"] = self._tfidf_vect is not None and self._tfidf_matrix is not None
            if self._tfidf_vect is not None and self._tfidf_matrix is not None:
                info["tfidf_matrix_shape"] = self._tfidf_matrix.shape
            info["embedder_available"] = self._embedder is not None
            
            # Sample some documents
            if self._simple_docs:
                info["sample_docs"] = [
                    {
                        "id": doc.get("id"),
                        "text_preview": doc.get("text", "")[:100] + "..." if len(doc.get("text", "")) > 100 else doc.get("text", ""),
                        "has_embedding": "emb" in doc and doc.get("emb"),
                        "meta_keys": list(doc.get("meta", {}).keys())
                    }
                    for doc in self._simple_docs[:3]
                ]
        
        return info

    def force_simple_mode(self) -> None:
        """Force the retriever to use simple mode."""
        logger.info("[PFU-RAG] Forcing simple mode")
        self._mode = "simple"
        self._client = None
        self._col = None
        self._ensure_backend()
        
        # Ensure TF-IDF is initialized if needed
        if self._simple_method == "tfidf" and self._simple_docs and (self._tfidf_vect is None or self._tfidf_matrix is None):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
                logger.info(f"[PFU-RAG] Building TF-IDF vectorizer for {len(self._simple_docs)} documents")
                texts = [rec.get("text", "") for rec in self._simple_docs]
                self._tfidf_vect = TfidfVectorizer(max_features=50000)
                self._tfidf_matrix = self._tfidf_vect.fit_transform(texts)
                logger.info(f"[PFU-RAG] TF-IDF vectorizer built successfully, matrix shape: {self._tfidf_matrix.shape}")
            except Exception as e:
                logger.error(f"[PFU-RAG] Failed to build TF-IDF vectorizer: {e}")
                # Don't raise here, just log the error and continue
                self._tfidf_vect = None
                self._tfidf_matrix = None

    def reload_documents(self) -> int:
        """Force reload documents from the simple store file."""
        if self._mode != "simple":
            logger.warning("[PFU-RAG] Cannot reload documents in non-simple mode")
            return 0
        
        logger.info(f"[PFU-RAG] Reloading documents from {self._simple_store_path}")
        if not os.path.isfile(self._simple_store_path):
            logger.warning(f"[PFU-RAG] Simple store file does not exist: {self._simple_store_path}")
            self._simple_docs = []
            return 0
        
        try:
            with open(self._simple_store_path, "r", encoding="utf-8") as fh:
                lines = [line.strip() for line in fh if line.strip()]
                logger.info(f"[PFU-RAG] Found {len(lines)} non-empty lines in file")
                self._simple_docs = [json.loads(line) for line in lines]
                logger.info(f"[PFU-RAG] Successfully reloaded {len(self._simple_docs)} documents")
                
                # Log some sample documents for debugging
                if self._simple_docs:
                    logger.info(f"[PFU-RAG] Sample document keys: {list(self._simple_docs[0].keys())}")
                    logger.debug(f"[PFU-RAG] First document: {self._simple_docs[0]}")
                
                # Reinitialize TF-IDF vectorizer if needed
                if self._simple_method == "tfidf" and self._simple_docs:
                    try:
                        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
                        logger.info(f"[PFU-RAG] Rebuilding TF-IDF vectorizer for {len(self._simple_docs)} reloaded documents")
                        texts = [rec.get("text", "") for rec in self._simple_docs]
                        self._tfidf_vect = TfidfVectorizer(max_features=50000)
                        self._tfidf_matrix = self._tfidf_vect.fit_transform(texts)
                        logger.info(f"[PFU-RAG] TF-IDF vectorizer rebuilt successfully, matrix shape: {self._tfidf_matrix.shape}")
                    except Exception as e:
                        logger.error(f"[PFU-RAG] Failed to rebuild TF-IDF vectorizer: {e}")
                        # Don't raise here, just log the error and continue
                        self._tfidf_vect = None
                        self._tfidf_matrix = None
                
                return len(self._simple_docs)
        except Exception as e:
            logger.error(f"[PFU-RAG] Failed to reload documents: {e}")
            self._simple_docs = []
            return 0


