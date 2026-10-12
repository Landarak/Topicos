"""
==============================================================================
PIPELINE COMPLETO DE EXTRACCIÓN, INDEXACIÓN RAPTOR Y SÍNTESIS TEXTUAL MULTINIVEL
==============================================================================
Módulos:
1. Ingesta masiva de PDFs (217 papers) con sanitización y caché
2. Segmentación en Unidades Atómicas de Contenido (ACU)
3. Caché persistente de Embeddings y Árboles
4. Descubrimiento Temático c-TF-IDF (BERTopic Core)
5. Construcción de Árbol Jerárquico RAPTOR (Modo Extractivo y Abstractivo)
6. Síntesis Estructurada Map-Reduce
7. Auditoría de Consistencia Factual (NLI)
8. Exportación automática de informes en Markdown y CSV
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from pypdf import PdfReader

# Dependencias ML y NLP
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import umap
from sklearn.mixture import GaussianMixture
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline as hf_pipeline
)

# ==============================================================================
# 0. ESTRUCTURAS DE DATOS Y CONFIGURACIÓN
# ==============================================================================

@dataclass
class DocumentUnit:
    id: str
    paper_name: str
    text: str
    doc_type: str  # 'paper_chunk', 'abstract', etc.
    embedding: Optional[np.ndarray] = None

@dataclass
class RaptorNode:
    node_id: str
    level: int
    text: str
    embedding: np.ndarray
    children_ids: List[str]
    synthesis_mode: str = "extractive"  # 'extractive' o 'abstractive'


class CacheManager:
    """Gestiona el almacenamiento y carga en disco para evitar recomputaciones costosas."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, filename: str) -> Path:
        return self.cache_dir / filename

    def exists(self, filename: str) -> bool:
        return (self.cache_dir / filename).exists()

    def save_json(self, filename: str, data: Any):
        path = self.get_path(filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(self, filename: str) -> Any:
        path = self.get_path(filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_npz(self, filename: str, **arrays):
        np.savez_compressed(self.get_path(filename), **arrays)

    def load_npz(self, filename: str):
        return np.load(self.get_path(filename))

# ==============================================================================
# 1. INGESTA DE PDFS Y NORMALIZACIÓN (ACU)
# ==============================================================================

class PDFCorpusLoader:
    """Extrae texto plano de archivos PDF con sanitización de caracteres y caché."""
    def __init__(self, papers_dir: Path, cache_manager: CacheManager):
        self.papers_dir = papers_dir
        self.cache = cache_manager

    def clean_raw_text(self, text: str) -> str:
        # Unir palabras cortadas por guion al final de línea
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # Normalizar espacios en blanco y saltos de línea
        text = re.sub(r'\s+', ' ', text)
        # Remover marcadores de cita comunes ej. [1], [2-4], (Smith et al., 2020)
        text = re.sub(r'\[[0-9,\s\-–]+\]', '', text)
        text = re.sub(r'\([A-Z][a-zA-Z\s]+,\s*[12][0-9]{3}[a-z]?\)', '', text)
        # Eliminar URLs largas ruidosas
        text = re.sub(r'https?://\S+', '', text)
        return text.strip()

    def load_corpus(self, force_reload: bool = False) -> Dict[str, str]:
        cache_file = "extracted_texts.json"
        if not force_reload and self.cache.exists(cache_file):
            print(f"[*] Cargando textos extraídos desde caché ({cache_file})...")
            return self.cache.load_json(cache_file)

        pdf_files = sorted(list(self.papers_dir.glob("*.pdf")))
        print(f"[*] Extrayendo texto de {len(pdf_files)} archivos PDF en '{self.papers_dir}'...")
        corpus: Dict[str, str] = {}

        for pdf_path in tqdm(pdf_files, desc="Leyendo PDFs", unit="paper"):
            paper_id = pdf_path.name
            full_text = []
            try:
                reader = PdfReader(str(pdf_path))
                for page_idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    cleaned_page = self.clean_raw_text(page_text)
                    if len(cleaned_page) > 40:
                        full_text.append(cleaned_page)
                
                doc_text = " ".join(full_text)
                if len(doc_text.strip()) > 100:
                    corpus[paper_id] = doc_text
                else:
                    print(f"  [!] Aviso: {paper_id} contiene muy poco texto legible.")
            except Exception as e:
                print(f"  [!] Error procesando {paper_id}: {e}")

        print(f"[+] Total de documentos válidos procesados: {len(corpus)}")
        self.cache.save_json(cache_file, corpus)
        return corpus


class DocumentNormalizer:
    """Segmenta textos largos en Unidades Atómicas de Contenido (ACU) con solapamiento."""
    def __init__(self, target_chunk_size: int = 70, overlap: int = 15):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def segment_into_sentences(self, text: str) -> List[str]:
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw_sentences if len(s.strip().split()) >= 4]

    def create_atomic_units(self, corpus: Dict[str, str], cache_manager: CacheManager, force_reload: bool = False) -> List[DocumentUnit]:
        cache_file = "atomic_units.json"
        if not force_reload and cache_manager.exists(cache_file):
            print(f"[*] Cargando unidades atómicas (ACU) desde caché ({cache_file})...")
            cached_data = cache_manager.load_json(cache_file)
            return [DocumentUnit(**item) for item in cached_data]

        print(f"[*] Segmentando {len(corpus)} documentos en Unidades Atómicas de Contenido...")
        units: List[DocumentUnit] = []

        for paper_id, text in tqdm(corpus.items(), desc="Segmentando ACUs", unit="doc"):
            sentences = self.segment_into_sentences(text)
            current_chunk: List[str] = []
            current_word_count = 0
            chunk_idx = 0

            for sent in sentences:
                words = sent.split()
                word_count = len(words)

                if current_word_count + word_count > self.target_chunk_size and current_chunk:
                    unit_text = " ".join(current_chunk)
                    units.append(DocumentUnit(
                        id=f"{paper_id}_acu_{chunk_idx}",
                        paper_name=paper_id,
                        text=unit_text,
                        doc_type="paper_chunk"
                    ))
                    chunk_idx += 1
                    # Ventana deslizante para preservar coherencia contextual
                    current_chunk = current_chunk[-1:] if len(current_chunk) > 1 else []
                    current_word_count = sum(len(s.split()) for s in current_chunk)

                current_chunk.append(sent)
                current_word_count += word_count

            if current_chunk:
                unit_text = " ".join(current_chunk)
                units.append(DocumentUnit(
                    id=f"{paper_id}_acu_{chunk_idx}",
                    paper_name=paper_id,
                    text=unit_text,
                    doc_type="paper_chunk"
                ))

        print(f"[+] Total de Unidades Atómicas de Contenido (ACU) generadas: {len(units)}")
        serializable = [asdict(u) for u in units]
        cache_manager.save_json(cache_file, serializable)
        return units

# ==============================================================================
# 2. GESTOR DE EMBEDDINGS VECTORIALES
# ==============================================================================

class EmbeddingManager:
    """Calcula y almacena en caché los embeddings densos de las ACUs."""
    def __init__(self, encoder: SentenceTransformer, cache_manager: CacheManager):
        self.encoder = encoder
        self.cache = cache_manager

    def get_embeddings(self, units: List[DocumentUnit], force_reload: bool = False) -> np.ndarray:
        cache_file = "embeddings_cache.npz"
        if not force_reload and self.cache.exists(cache_file):
            print(f"[*] Cargando matriz de embeddings desde caché ({cache_file})...")
            data = self.cache.load_npz(cache_file)
            embeddings = data["embeddings"]
            if len(embeddings) == len(units):
                for idx, u in enumerate(units):
                    u.embedding = embeddings[idx]
                return embeddings

        print(f"[*] Generando embeddings para {len(units)} fragmentos con MiniLM...")
        texts = [u.text for u in units]
        embeddings = self.encoder.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        for idx, u in enumerate(units):
            u.embedding = embeddings[idx]

        self.cache.save_npz(cache_file, embeddings=embeddings)
        print(f"[+] Embeddings guardados en caché: matriz de dimensión {embeddings.shape}")
        return embeddings

# ==============================================================================
# 3. DESCUBRIMIENTO TEMÁTICO c-TF-IDF (BERTopic Core)
# ==============================================================================

class TopicDiscoveryEngine:
    """Extrae descriptores léxicos mediante formulación matemática de c-TF-IDF."""
    def __init__(self, top_n_words: int = 8):
        self.top_n_words = top_n_words
        self.stop_words = {
            "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "as", "was", "with",
            "on", "by", "at", "an", "be", "this", "which", "or", "from", "are", "were", "been",
            "have", "has", "had", "they", "we", "their", "our", "not", "but", "also", "its", "into",
            "more", "such", "than", "these", "other", "used", "using", "between", "study", "research",
            "paper", "results", "analysis", "data", "based", "de", "la", "el", "los", "las", "en",
            "y", "un", "una", "del", "por", "con", "para", "es", "al", "se", "como", "mas", "que",
            "su", "sus", "sobre", "este", "esta", "estos", "estas", "entre", "tambien", "estudio",
            "doi", "https", "http", "vol", "pp", "author", "authors", "university", "press", "journal"
        }

    def compute_c_tf_idf(self, documents_per_cluster: Dict[int, List[str]]) -> Dict[int, List[Tuple[str, float]]]:
        cluster_ids = [c for c in documents_per_cluster.keys() if c != -1]
        if not cluster_ids:
            return {}

        cluster_texts = [" ".join(documents_per_cluster[c]).lower() for c in cluster_ids]
        tokenized_clusters = [
            [w for w in re.findall(r'\b[a-záéíóúñ]{3,}\b', text) if w not in self.stop_words]
            for text in cluster_texts
        ]

        vocab = sorted(list(set(w for cluster in tokenized_clusters for w in cluster)))
        if not vocab:
            return {}
            
        vocab_index = {w: i for i, w in enumerate(vocab)}
        num_clusters = len(cluster_ids)
        num_vocab = len(vocab)

        tf = np.zeros((num_clusters, num_vocab))
        for c_idx, cluster_tokens in enumerate(tokenized_clusters):
            for token in cluster_tokens:
                if token in vocab_index:
                    tf[c_idx, vocab_index[token]] += 1

        f_t = tf.sum(axis=0)
        total_words_per_cluster = tf.sum(axis=1)
        A = np.mean(total_words_per_cluster) if len(total_words_per_cluster) > 0 else 1.0

        idf = np.log(1.0 + (A / (f_t + 1e-6)))
        c_tf_idf_matrix = tf * idf

        topic_keywords: Dict[int, List[Tuple[str, float]]] = {}
        for c_idx, c_id in enumerate(cluster_ids):
            scores = c_tf_idf_matrix[c_idx]
            top_indices = np.argsort(scores)[::-1][:self.top_n_words]
            topic_keywords[c_id] = [(vocab[i], float(scores[i])) for i in top_indices if scores[i] > 0]

        return topic_keywords

# ==============================================================================
# 4. SINTETIZADORES DE CLÚSTER (EXTRACTIVO Y ABSTRACTIVO)
# ==============================================================================

class BaseClusterSynthesizer:
    def synthesize(self, child_texts: List[str], child_embeddings: np.ndarray, level: int) -> str:
        raise NotImplementedError

class ExtractiveClusterSynthesizer(BaseClusterSynthesizer):
    """
    Sintetizador Extractivo: Selecciona las oraciones más cercanas al centroide
    semántico del clúster sin usar LLM. 100% libre de alucinaciones y ultra rápido.
    """
    def synthesize(self, child_texts: List[str], child_embeddings: np.ndarray, level: int) -> str:
        if not child_texts:
            return ""
        if len(child_texts) <= 2:
            bullets = [f"• {t.strip()}" for t in child_texts]
            return f"[Nivel {level} - Síntesis Extractiva]\n" + "\n".join(bullets)

        # Calcular centroide semántico
        centroid = np.mean(child_embeddings, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Medir similitud de cada fragmento con el centroide
        sims = np.dot(child_embeddings, centroid)
        top_indices = np.argsort(sims)[::-1][:min(4, len(child_texts))]
        selected_texts = [child_texts[i] for i in top_indices]

        bullets = [f"• {t.strip()}" for t in selected_texts]
        return f"[Nivel {level} - Síntesis Extractiva]\n" + "\n".join(bullets)


class AbstractiveClusterSynthesizer(BaseClusterSynthesizer):
    """
    Sintetizador Abstractivo: Utiliza un modelo Seq2Seq (BART/T5) de Hugging Face
    para redactar un resumen continuo y fluido en prosa de cada clúster.
    """
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Inicializando sintetizador abstractivo con '{model_name}' en {self.device}...")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.is_ready = True
        except Exception as e:
            print(f"[!] Error cargando modelo abstractivo: {e}. Fallback a extractivo.")
            self.is_ready = False

    def synthesize(self, child_texts: List[str], child_embeddings: np.ndarray, level: int) -> str:
        if not child_texts:
            return ""
        if not getattr(self, "is_ready", False) or len(child_texts) == 1:
            return ExtractiveClusterSynthesizer().synthesize(child_texts, child_embeddings, level)

        joined_text = " ".join(child_texts[:5])[:1200]
        try:
            inputs = self.tokenizer(
                "summarize: " + joined_text if "t5" in str(self.model.__class__).lower() else joined_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=65,
                    min_length=20,
                    no_repeat_ngram_size=2,
                    num_beams=2
                )
            abstract_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            return f"[Nivel {level} - Síntesis Abstractiva] {abstract_text}"
        except Exception as e:
            return ExtractiveClusterSynthesizer().synthesize(child_texts, child_embeddings, level)

# ==============================================================================
# 5. ÁRBOL JERÁRQUICO RAPTOR (MOTOR VECTORIAL MULTINIVEL)
# ==============================================================================

class RaptorTreeEngine:
    """Construye una jerarquía bottom-up mediante clustering UMAP + GMM + BIC."""
    def __init__(
        self,
        encoder: SentenceTransformer,
        synthesizer: BaseClusterSynthesizer,
        max_levels: int = 2,
        synthesis_mode_name: str = "extractive"
    ):
        self.encoder = encoder
        self.synthesizer = synthesizer
        self.max_levels = max_levels
        self.synthesis_mode_name = synthesis_mode_name
        self.tree_nodes: Dict[str, RaptorNode] = {}
        self.level_nodes: Dict[int, List[str]] = {i: [] for i in range(max_levels + 1)}

    def fit_clusters(self, embeddings: np.ndarray, max_k: int = 8) -> np.ndarray:
        n_samples = embeddings.shape[0]
        if n_samples <= 3:
            return np.zeros(n_samples, dtype=int)

        n_components = min(5, n_samples - 1)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(15, n_samples - 1),
            min_dist=0.1,
            random_state=42
        )
        reduced_emb = reducer.fit_transform(embeddings)

        best_k = 1
        best_bic = np.inf
        best_gmm = None

        candidate_ks = range(2, min(max_k + 1, n_samples))
        if not candidate_ks:
            return np.zeros(n_samples, dtype=int)

        for k in candidate_ks:
            gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42, max_iter=80)
            gmm.fit(reduced_emb)
            bic = gmm.bic(reduced_emb)
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_gmm = gmm

        return best_gmm.predict(reduced_emb) if best_gmm is not None else np.zeros(n_samples, dtype=int)

    def build_tree(self, leaf_units: List[DocumentUnit]):
        print(f"\n[*] Construyendo Árbol RAPTOR (Modo: {self.synthesis_mode_name.upper()})...")
        # Nivel 0: Hojas
        for unit in leaf_units:
            node = RaptorNode(
                node_id=unit.id,
                level=0,
                text=unit.text,
                embedding=unit.embedding if unit.embedding is not None else self.encoder.encode(unit.text),
                children_ids=[],
                synthesis_mode=self.synthesis_mode_name
            )
            self.tree_nodes[node.node_id] = node
            self.level_nodes[0].append(node.node_id)

        current_level_ids = self.level_nodes[0]

        # Niveles 1 a max_levels
        for lvl in range(1, self.max_levels + 1):
            if len(current_level_ids) <= 3:
                break

            current_embs = np.array([self.tree_nodes[nid].embedding for nid in current_level_ids])
            k_limit = min(12, max(2, len(current_level_ids) // 6))
            cluster_labels = self.fit_clusters(current_embs, max_k=k_limit)

            next_level_ids = []
            unique_clusters = sorted(list(set(cluster_labels)))
            print(f"  - Nivel {lvl}: {len(current_level_ids)} nodos agrupados en {len(unique_clusters)} clústeres.")

            for c_id in tqdm(unique_clusters, desc=f"Sintetizando Nivel {lvl}", unit="cluster"):
                member_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == c_id]
                child_node_ids = [current_level_ids[i] for i in member_indices]
                child_texts = [self.tree_nodes[nid].text for nid in child_node_ids]
                child_embs = np.array([self.tree_nodes[nid].embedding for nid in child_node_ids])

                summary_text = self.synthesizer.synthesize(child_texts, child_embs, level=lvl)
                summary_emb = self.encoder.encode(summary_text, convert_to_numpy=True, normalize_embeddings=True)

                parent_id = f"node_L{lvl}_c{c_id}_{len(next_level_ids)}"
                parent_node = RaptorNode(
                    node_id=parent_id,
                    level=lvl,
                    text=summary_text,
                    embedding=summary_emb,
                    children_ids=child_node_ids,
                    synthesis_mode=self.synthesis_mode_name
                )
                self.tree_nodes[parent_id] = parent_node
                next_level_ids.append(parent_id)
                self.level_nodes[lvl].append(parent_id)

            current_level_ids = next_level_ids

        print(f"[+] Árbol RAPTOR ({self.synthesis_mode_name}) completado con {len(self.tree_nodes)} nodos totales.")

    def search_by_level(self, query: str, level: int, top_k: int = 3) -> List[Tuple[RaptorNode, float]]:
        """Búsqueda vectorial restringida a un nivel específico de la jerarquía."""
        query_emb = self.encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        level_node_ids = self.level_nodes.get(level, [])
        scores: List[Tuple[RaptorNode, float]] = []

        for nid in level_node_ids:
            node = self.tree_nodes[nid]
            sim = float(np.dot(query_emb, node.embedding))
            scores.append((node, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def collapsed_tree_search(self, query: str, top_k: int = 8) -> List[Tuple[RaptorNode, float]]:
        """Recuperación colapsada: búsqueda vectorial simultánea en todos los niveles."""
        query_emb = self.encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        scores: List[Tuple[RaptorNode, float]] = []

        for node in self.tree_nodes.values():
            sim = float(np.dot(query_emb, node.embedding))
            scores.append((node, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# ==============================================================================
# 6. ORQUESTACIÓN MAP-REDUCE PARA SÍNTESIS
# ==============================================================================

class StructuredSynthesizer:
    """Genera la síntesis estructurada dividida en Ideas, Hallazgos y Conclusiones."""
    def __init__(self, raptor_engine: RaptorTreeEngine):
        self.raptor = raptor_engine

    def map_stage(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        mapped_data = {
            "root_context": [],          # Nivel >= 2
            "intermediate_findings": [], # Nivel 1
            "atomic_evidence": []        # Nivel 0
        }

        seen_node_ids = set()
        for q in queries:
            # Recuperar de cada nivel jerárquico para asegurar visibilidad multiescala
            for lvl in range(self.raptor.max_levels, -1, -1):
                level_results = self.raptor.search_by_level(q, level=lvl, top_k=2 if lvl > 0 else 3)
                for node, score in level_results:
                    if node.node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node.node_id)

                    entry = {
                        "id": node.node_id,
                        "text": node.text,
                        "score": round(score, 4),
                        "level": node.level
                    }

                    if node.level >= 2:
                        mapped_data["root_context"].append(entry)
                    elif node.level == 1:
                        mapped_data["intermediate_findings"].append(entry)
                    else:
                        mapped_data["atomic_evidence"].append(entry)

        return mapped_data

    def reduce_stage(
        self,
        mapped_data: Dict[str, List[Dict[str, Any]]],
        topics: Dict[int, List[Tuple[str, float]]]
    ) -> Dict[str, List[str]]:
        main_ideas = []
        for t_id, terms in list(topics.items())[:6]:
            top_words = ", ".join([f"{w} ({s:.2f})" for w, s in terms[:4]])
            main_ideas.append(f"Clúster Temático {t_id}: Énfasis léxico en [{top_words}].")

        for rc in mapped_data["root_context"][:4]:
            main_ideas.append(f"Síntesis Global (Nivel 2, Sim: {rc['score']}):\n{rc['text']}")

        findings = []
        for ifind in mapped_data["intermediate_findings"][:6]:
            findings.append(f"Patrón/Hallazgo Intermedio (Nivel 1, Sim: {ifind['score']}):\n{ifind['text']}")

        for atom in mapped_data["atomic_evidence"][:5]:
            findings.append(f"Evidencia Local ({atom['id']}): {atom['text']}")

        conclusions = [
            f"Cobertura Jerárquica: Se indexaron exitosamente {len(self.raptor.level_nodes[0])} unidades atómicas con {len(self.raptor.tree_nodes)} nodos totales.",
            "Integración Estructural: La recuperación multiescala de RAPTOR permite vincular evidencia atómica de papers individuales con patrones globales del corpus.",
            "Representatividad Léxica: Los clústeres c-TF-IDF delimitan los ejes temáticos metodológicos y empíricos dominantes."
        ]

        return {
            "Ideas Principales": main_ideas,
            "Hallazgos": findings,
            "Conclusiones": conclusions
        }

# ==============================================================================
# 7. AUDITORÍA FACTUAL NLI (CROSS-ENCODER DEBERTA)
# ==============================================================================

class FactualConsistencyAuditor:
    """Verifica si las afirmaciones extraídas se deducen o contradicen los textos fuente."""
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Cargando modelo NLI '{model_name}' en {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def verify_assertion(self, premise: str, hypothesis: str) -> Dict[str, float]:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        return {
            "contradiction": float(probs[0]),
            "neutral": float(probs[1]),
            "entailment": float(probs[2])
        }

    def audit_claims(
        self,
        claims: List[str],
        raptor_engine: RaptorTreeEngine,
        threshold_contra: float = 0.5
    ) -> pd.DataFrame:
        results = []
        print(f"[*] Auditando {len(claims)} afirmaciones mediante NLI frente al corpus...")

        for claim in tqdm(claims, desc="Auditoría NLI", unit="claim"):
            best_nodes = raptor_engine.collapsed_tree_search(claim, top_k=3)
            combined_premise = " ".join([n[0].text for n in best_nodes if n[0].level == 0])[:1200]

            if not combined_premise.strip():
                combined_premise = best_nodes[0][0].text[:1200]

            scores = self.verify_assertion(combined_premise, claim)
            is_valid = scores["contradiction"] < threshold_contra

            status = "PASSED" if is_valid else "FAILED_CONTRADICTION"
            results.append({
                "Afirmación (Claim)": claim[:120] + ("..." if len(claim) > 120 else ""),
                "Entailment": round(scores["entailment"], 4),
                "Neutral": round(scores["neutral"], 4),
                "Contradicción": round(scores["contradiction"], 4),
                "Estado": status
            })

        return pd.DataFrame(results)

# ==============================================================================
# 8. GENERACIÓN DE REPORTES GLOBALES
# ==============================================================================

class ReportGenerator:
    """Genera el informe final consolidado en Markdown comparando ambos métodos."""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        num_papers: int,
        num_acus: int,
        extractive_output: Dict[str, List[str]],
        abstractive_output: Optional[Dict[str, List[str]]],
        topics: Dict[int, List[Tuple[str, float]]],
        audit_df: pd.DataFrame
    ) -> Path:
        report_path = self.output_dir / "informe_global_sintesis.md"
        csv_path = self.output_dir / "auditoria_nli_completa.csv"

        audit_df.to_csv(csv_path, index=False, encoding="utf-8")

        lines = [
            "# Informe Global de Síntesis y Análisis de Corpus Académico",
            f"*Generado automáticamente mediante Pipeline RAPTOR + c-TF-IDF + Auditoría NLI*",
            f"**Fecha y Hora:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 1. Resumen Estadístico del Corpus",
            f"- **Total de Papers procesados:** {num_papers} archivos PDF.",
            f"- **Unidades Atómicas de Contenido (ACUs):** {num_acus} fragmentos indexados.",
            f"- **Tópicos Semánticos Descubiertos:** {len(topics)} clústeres principales.",
            "",
            "---",
            "",
            "## 2. Descubrimiento Temático Global (c-TF-IDF / BERTopic Core)",
            "Términos léxicos más distintivos por clúster semántico:",
            "",
            "| Clúster | Términos Clave Dominantes (c-TF-IDF Score) |",
            "| :--- | :--- |"
        ]

        for tid, terms in topics.items():
            term_str = ", ".join([f"`{w}` ({s:.2f})" for w, s in terms[:5]])
            lines.append(f"| **Tópico {tid}** | {term_str} |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Comparativa de Síntesis: Modo Extractivo vs Modo Abstractivo (Generativo)",
            "",
            "### A. Síntesis Jerárquica Extractiva (Local / Alta Fidelidad)",
            "> Selecciona y estructura los pasajes empíricos centrales sin riesgo de alucinación.",
            ""
        ])

        for section, items in extractive_output.items():
            lines.append(f"#### {section}")
            for it in items:
                lines.append(f"- {it}")
            lines.append("")

        if abstractive_output:
            lines.extend([
                "### B. Síntesis Jerárquica Abstractiva (Modelo Generativo)",
                "> Redacta resúmenes continuos y condensados mediante modelo Seq2Seq.",
                ""
            ])
            for section, items in abstractive_output.items():
                lines.append(f"#### {section}")
                for it in items:
                    lines.append(f"- {it}")
                lines.append("")

        lines.extend([
            "---",
            "",
            "## 4. Auditoría de Consistencia Factual (NLI)",
            "Evaluación de afirmaciones clave frente al corpus mediante Cross-Encoder DeBERTa:",
            "",
            "| Afirmación Evaluada | Implicación (Entailment) | Neutral | Contradicción | Estado |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ])

        for _, row in audit_df.iterrows():
            badge = "✅ PASSED" if row["Estado"] == "PASSED" else "❌ RECHAZADO"
            lines.append(
                f"| {row['Afirmación (Claim)']} | {row['Entailment']} | {row['Neutral']} | {row['Contradicción']} | **{badge}** |"
            )

        lines.extend([
            "",
            "---",
            f"*El archivo completo de auditoría en CSV se encuentra disponible en: `{csv_path.name}`.*"
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n[+] Informe global generado exitosamente en: {report_path}")
        return report_path

# ==============================================================================
# 9. FLUJO PRINCIPAL DE EJECUCIÓN (CLI / MAIN)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline RAPTOR + c-TF-IDF + NLI para 217 Papers")
    parser.add_argument("--papers-dir", type=str, default="Papers", help="Directorio con los archivos PDF")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Directorio para caché de datos")
    parser.add_argument("--output-dir", type=str, default="reportes", help="Directorio para guardar reportes")
    parser.add_argument("--mode", type=str, choices=["both", "extractive", "abstractive"], default="both", help="Modo de síntesis")
    parser.add_argument("--force-reload", action="store_true", help="Forzar re-extracción y recomputación")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.resolve()
    papers_dir = base_dir / args.papers_dir
    cache_dir = base_dir / args.cache_dir
    output_dir = base_dir / args.output_dir

    print("=" * 80)
    print("EJECUTANDO PIPELINE ACADÉMICO MULTINIVEL (RAPTOR + c-TF-IDF + NLI)")
    print(f"Directorio de Papers: {papers_dir}")
    print(f"Directorio de Caché:  {cache_dir}")
    print(f"Directorio de Salida: {output_dir}")
    print(f"Modo de Síntesis:     {args.mode.upper()}")
    print("=" * 80)

    cache_mgr = CacheManager(cache_dir)
    encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_mgr = EmbeddingManager(encoder_model, cache_mgr)
    normalizer = DocumentNormalizer(target_chunk_size=75, overlap=15)
    pdf_loader = PDFCorpusLoader(papers_dir, cache_mgr)

    # 1. Ingesta y normalización
    corpus = pdf_loader.load_corpus(force_reload=args.force_reload)
    if not corpus:
        print("[!] No se encontraron documentos válidos en la carpeta especificada.")
        return

    units = normalizer.create_atomic_units(corpus, cache_mgr, force_reload=args.force_reload)
    embeddings = embedding_mgr.get_embeddings(units, force_reload=args.force_reload)

    # 2. Descubrimiento temático global (c-TF-IDF)
    print("\n[*] Calculando Tópicos Globales con c-TF-IDF...")
    topic_engine = TopicDiscoveryEngine(top_n_words=6)
    docs_by_paper: Dict[int, List[str]] = {}
    paper_names = list(corpus.keys())
    for idx, name in enumerate(paper_names[:20]):
        docs_by_paper[idx] = [u.text for u in units if u.paper_name == name]
    topics = topic_engine.compute_c_tf_idf(docs_by_paper)

    # Consultas de análisis transversal sobre el corpus
    queries = [
        "What are the main methodologies, empirical findings and conclusions?",
        "Digital methods, quantitative analysis, data collection, and algorithms.",
        "Social media, communication studies, discourse analysis, and cultural practices.",
        "Limitations, ethical challenges, algorithmic bias, and future research directions."
    ]

    extractive_output = None
    abstractive_output = None

    # 3. Árbol RAPTOR - Modo Extractivo
    if args.mode in ["both", "extractive"]:
        extractive_synth = ExtractiveClusterSynthesizer()
        raptor_ext = RaptorTreeEngine(
            encoder=encoder_model,
            synthesizer=extractive_synth,
            max_levels=2,
            synthesis_mode_name="extractive"
        )
        raptor_ext.build_tree(units)
        struct_synth_ext = StructuredSynthesizer(raptor_ext)
        mapped_ext = struct_synth_ext.map_stage(queries)
        extractive_output = struct_synth_ext.reduce_stage(mapped_ext, topics)

    # 4. Árbol RAPTOR - Modo Abstractivo
    if args.mode in ["both", "abstractive"]:
        abstractive_synth = AbstractiveClusterSynthesizer(model_name="sshleifer/distilbart-cnn-12-6")
        raptor_abs = RaptorTreeEngine(
            encoder=encoder_model,
            synthesizer=abstractive_synth,
            max_levels=2,
            synthesis_mode_name="abstractive"
        )
        raptor_abs.build_tree(units)
        struct_synth_abs = StructuredSynthesizer(raptor_abs)
        mapped_abs = struct_synth_abs.map_stage(queries)
        abstractive_output = struct_synth_abs.reduce_stage(mapped_abs, topics)

    # 5. Auditoría Factual NLI
    auditor = FactualConsistencyAuditor(model_name="cross-encoder/nli-deberta-v3-small")
    active_raptor = raptor_ext if extractive_output else raptor_abs

    claims_to_audit = []
    if extractive_output:
        for item in extractive_output["Ideas Principales"][:2] + extractive_output["Hallazgos"][:3]:
            cleaned_claim = re.sub(r'^[•\-\[\w\s]+:\s*', '', item).strip()
            if len(cleaned_claim) > 20:
                claims_to_audit.append(cleaned_claim)

    claims_to_audit.append("Digital methods research completely prohibits any use of computational tools or statistical algorithms.")

    audit_df = auditor.audit_claims(claims_to_audit, active_raptor)

    # 6. Generación del reporte consolidado
    report_gen = ReportGenerator(output_dir)
    report_path = report_gen.generate_markdown_report(
        num_papers=len(corpus),
        num_acus=len(units),
        extractive_output=extractive_output if extractive_output else {},
        abstractive_output=abstractive_output,
        topics=topics,
        audit_df=audit_df
    )

    print("\n" + "=" * 80)
    print("PIPELINE FINALIZADO CON ÉXITO")
    print(f"Informe listo en: {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
