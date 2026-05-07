import os
import json
import re
import datetime
import hashlib
import logging
from pathlib import Path

import numpy as np
import networkx as nx
import streamlit as st
from openai import OpenAI, BadRequestError, APIError, APITimeoutError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_METRICS_FILE = os.path.join(SCRIPT_DIR, "accepted_new_metrics.json")
EMBEDDING_CACHE_DIR = os.path.join(SCRIPT_DIR, ".kg_embedding_cache")
EMBEDDING_CACHE_VERSION = 2  
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_EMBED = 25
GRAPH_HOP_LIMIT = 2
MAX_CANDIDATES = 25
API_TIMEOUT = 120  

logger = logging.getLogger(__name__)

OVERLAP_RANK = {"duplicate": 4, "high": 3, "moderate": 2, "low": 1}

CLASSIFICATION_SCHEMA = {
    "analytical_dimension": {
        "label": "Analytical dimension",
        "predicate": "is_a",
        "description": "What type of QC metric this is (the metric's place in the taxonomy).",
        "values": {
            "acquisition coverage metric": "How comprehensively data were collected (scan counts, sampling density).",
            "mass accuracy metric": "Deviation between observed and theoretical m/z.",
            "intensity stability metric": "Variation of signal intensity over time.",
            "chromatographic performance metric": "Separation performance (peak width, symmetry, RT reproducibility).",
            "ionization quality metric": "Properties of the precursor ion population (charge-state distribution, adducts).",
            "ion mobility metric": "IMS resolution, drift-time/CCS accuracy and reproducibility.",
            "spectral quality metric": "Quality of individual spectra (peak density, S/N, completeness, entropy).",
            "fragmentation efficiency metric": "Effectiveness of precursor fragmentation to produce interpretable spectra.",
            "isolation purity metric": "Precursor isolation selectivity or co-isolation of interfering species.",
            "identification confidence metric": "Reliability of identifications (FDR, ID rate).",
            "quantification precision metric": "Reproducibility or variability of quantitative results.",
            "contamination metric": "Unwanted signal from contaminants, carryover, or background.",
            "instrument operational performance metric": "General instrument health (vacuum, detector voltage, temperature).",
            "missingness/completeness metric": "Data absence or completeness across features, runs, or studies.",
        },
    },
    "workflow_stage": {
        "label": "Workflow stage",
        "predicate": "part_of_workflow_stage",
        "description": (
            "Where in the experimental/computational pipeline the measured "
            "QUANTITY originates. This is about what physical or computational "
            "process generates the quantity, NOT about what input data is "
            "required to compute it."
        ),
        "values": {
            "sample preparation stage": "The quantity describes sample handling, labeling, digestion, or storage quality.",
            "chromatography stage": "The quantity describes LC separation performance (peak width, RT).",
            "ionization stage": "The quantity describes ion generation and charge distribution.",
            "ion mobility separation stage": "The quantity describes gas-phase separation device performance.",
            "mass spectrometry acquisition stage": "The quantity describes scanning, detection, or data acquisition (general, use when MS1/MS2 is ambiguous).",
            "MS1 acquisition stage": "The quantity is a property of MS1 data (precursor m/z, MS1 intensity, MS1 scans). Use this even if identification results are needed as a filter.",
            "MS2 acquisition stage": "The quantity is a property of MS2 data (fragment ions, MS2 spectra, isolation windows). Use this even if identification results are needed as a filter.",
            "MSn acquisition stage": "The quantity is a property of higher-order fragmentation (MS3, etc.).",
            "instrument performance monitoring stage": "The quantity describes general instrument health and stability.",
            "instrument calibration stage": "The quantity is derived from calibration routines or control samples.",
            "data preprocessing stage": "The quantity describes baseline correction, noise removal, or peak picking.",
            "identification stage": "The quantity is ITSELF an identification quality measure (FDR, ID rate, PSM count). Not for metrics that merely use IDs as a filter.",
            "quantification stage": "The quantity is ITSELF a quantitative accuracy or precision measure (CV of intensities, ratio reproducibility).",
            "integration stage": "The quantity describes alignment, normalization, or data integration across runs.",
            "environmental condition monitoring": "The quantity describes lab temperature, humidity, power fluctuations.",
        },
    },
    "data_dependency": {
        "label": "Information dependency",
        "predicate": "depends_on_data_type",
        "description": (
            "What type of input data the metric requires to be computed. "
            "This is separate from workflow_stage: a metric can measure an "
            "MS1 quantity (workflow_stage = MS1 acquisition stage) while "
            "requiring identification results as a filter "
            "(data_dependency = identification results)."
        ),
        "values": {
            "raw acquisition data": "Calculated directly from raw MS data without identifications or quantification.",
            "deconvoluted data": "Based on processed spectra or peak lists after deconvolution/centroiding, but before identification.",
            "identification results": "Depends on identified peptides, compounds, or spectra (even if only as a filter).",
            "quantification results": "Derived from quantitative data matrices.",
            "hybrid": "Combines multiple data types (e.g. identification and quantification).",
            "reference data": "Requires comparison to external standards or reference files (iRT peptides, calibration standards).",
        },
    },
    "measurement_scope": {
        "label": "Measurement scope",
        "predicate": "has_measurement_scope",
        "description": "At what aggregation level the metric summarizes data.",
        "values": {
            "spectrum level": "Per-spectrum metrics (one value per spectrum).",
            "pixel/voxel level": "Per-pixel metrics in imaging or spatial omics.",
            "feature level": "Per feature (peptide, compound, or chromatographic peak).",
            "run level": "Aggregated per LC-MS run (single summary value per run).",
            "batch level": "Aggregated across multiple related runs.",
            "study level": "Aggregated across an entire experiment or project.",
        },
    },
    "acquisition_mode": {
        "label": "Acquisition strategy",
        "predicate": "applies_to_acquisition_mode",
        "description": "Which acquisition mode or instrument configuration the metric is relevant for.",
        "values": {
            "acquisition mode independent": "Valid for any acquisition method (DDA, DIA, targeted, etc.).",
            "data-dependent acquisition (DDA)": "Specific to stochastic precursor selection workflows.",
            "data-independent acquisition (DIA)": "For window-based fragmentation strategies (SWATH, etc.).",
            "targeted acquisition": "For SRM, PRM, or other targeted workflows.",
            "ion-mobility-coupled metric": "Derived from acquisition methods including ion mobility separation.",
            "imaging acquisition": "For spatially resolved MS (MALDI, DESI, SIMS).",
            "other specialized mode": "Advanced or hybrid modes (BoxCar, MSn, multiplexed scanning).",
            "Orbitrap-specific": "Only applicable to Orbitrap instruments.",
            "TOF-specific": "Relevant to time-of-flight instruments.",
            "ion-trap-specific": "Specific to trap-based systems.",
            "other platform-specific": "For quadrupoles, FT-ICR, or hybrid systems.",
        },
    },
    "quality_directionality": {
        "label": "Quality interpretation",
        "predicate": "has_quality_directionality",
        "description": (
            "How the metric's numeric value relates to overall data quality. "
            "Choose based on the metric's direct relationship to quality."
        ),
        "values": {
            "higher is better": "Increasing values always indicate improved quality (identification rate, purity fraction, coverage, signal-to-noise).",
            "lower is better": "Decreasing values always indicate improved quality (FDR, absolute mass error, noise level, contamination fraction, number of empty scans).",
            "context dependent": "Interpretation genuinely varies by experimental design or method -- there is no single direction that is universally better (charge-state fractions, peak density, spectral entropy).",
            "target range": "Optimal quality corresponds to a specific value or interval, with deviations in either direction being bad (signed mass deviation centered on 0 ppm, temperature, pressure).",
            "categorical": "Quality expressed as discrete categories (pass/fail, OK/warning/error).",
            "trend": "Intended for temporal monitoring and drift detection rather than direct ranking (instrument drift over time, long-term TIC trend).",
        },
    },
    "metric_value_type": {
        "label": "Metric value type",
        "predicate": "has_value_type",
        "description": "The structural format of the metric's reported value(s) in mzQC.",
        "values": {
            "single value": "A single numeric or categorical value (e.g. one float, one integer, one string).",
            "tuple": "Several ordered values of the same kind (e.g. quantiles, quartile fractions).",
            "table": "Parallel lists of equal length; each column has its own unit.",
            "matrix": "2D array of homogeneous numeric values.",
        },
    },
}

CATEGORY_NAME_TO_ID = {
    "ID based metric": "MS:4000008",
    "ID free metric": "MS:4000009",
    "quantification based metric": "MS:4000010",
    "single run based metric": "MS:4000012",
    "multiple runs based metric": "MS:4000013",
    "single spectrum based metric": "MS:4000014",
    "multiple spectra based metric": "MS:4000015",
    "retention time metric": "MS:4000016",
    "chromatogram metric": "MS:4000017",
    "XIC metric": "MS:4000018",
    "MS metric": "MS:4000019",
    "ion source metric": "MS:4000020",
    "MS1 metric": "MS:4000021",
    "MS2 metric": "MS:4000022",
    "sample preparation metric": "MS:4000023",
    "environment metric": "MS:4000024",
    "QC sample metric": "MS:4000073",
    "QC2 sample metric": "MS:4000076",
    "total ion current chromatogram": "MS:1000235",
    "isolation window attribute": "MS:1000792",
}

VTYPE_NAME_TO_ID = {
    "single value": "MS:4000003",
    "n-tuple": "MS:4000004",
    "tuple": "MS:4000004",
    "table": "MS:4000005",
    "matrix": "MS:4000006",
}

UNIT_NAME_TO_ID = {
    "parts per million": "UO:0000169",
    "count unit": "UO:0000189",
    "second": "UO:0000010",
    "fraction": "UO:0000191",
    "dalton": "UO:0000221",
    "percent": "UO:0000187",
    "intensity unit": "MS:1000043",
    "pressure unit": "UO:0000109",
    "hertz": "UO:0000106",
    "electronvolt": "UO:0000266",
    "millisecond": "UO:0000028",
    "minute": "UO:0000031",
    "thompson": "UO:0000240",
}

# OBO parsing

def parse_obo(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    raw_blocks = re.split(r"\n(?=\[Term\])", text)
    terms = []
    for block in raw_blocks:
        if not block.strip().startswith("[Term]"):
            continue
        term = {
            "id": None, "name": None, "def": None, "def_xrefs": [],
            "comment": None, "synonyms": [], "is_a": [],
            "is_obsolete": False, "replaced_by": None, "categories": [],
            "units": [], "xsd_value_type": None, "value_concepts": [],
            "columns": [], "optional_columns": [], "relations": [],
            "order": None, "domain": None,
        }
        for line in block.strip().splitlines():
            line = line.strip()
            if not line or line == "[Term]":
                continue
            if line.startswith("id: "):
                term["id"] = line[4:].strip()
            elif line.startswith("name: "):
                term["name"] = line[6:].strip()
            elif line.startswith("def: "):
                m = re.match(r'def:\s*"(.*?)"\s*\[([^\]]*)\]', line)
                if m:
                    term["def"] = m.group(1)
                    term["def_xrefs"] = [r.strip() for r in m.group(2).split(",") if r.strip()]
            elif line.startswith("comment: "):
                term["comment"] = line[9:].strip()
            elif line.startswith("synonym: "):
                m = re.match(r'synonym:\s*"(.*?)"', line)
                if m:
                    term["synonyms"].append(m.group(1))
            elif line.startswith("is_a: "):
                term["is_a"].append(line[6:].strip())
            elif line == "is_obsolete: true":
                term["is_obsolete"] = True
            elif line.startswith("replaced_by: "):
                term["replaced_by"] = line[13:].strip()
            elif line.startswith("relationship: "):
                _parse_relationship(term, line[14:].strip())
        if term["id"] and term["name"]:
            terms.append(term)
    return terms


def _parse_relationship(term, rel):
    rel_map = {
        "has_metric_category ": "categories",
        "has_units ": "units",
        "has_value_concept ": "value_concepts",
        "has_column ": "columns",
        "has_optional_column ": "optional_columns",
        "has_relation ": "relations",
    }
    for prefix, key in rel_map.items():
        if rel.startswith(prefix):
            term[key].append(rel[len(prefix):].strip())
            return
    if rel.startswith("has_value_type "):
        term["xsd_value_type"] = rel[15:].strip()
    elif rel.startswith("has_order "):
        term["order"] = rel[10:].strip()
    elif rel.startswith("has_domain "):
        term["domain"] = rel[11:].strip()

# Knowledge graph

def build_knowledge_graph(raw_terms):
    G = nx.DiGraph()
    for t in raw_terms:
        G.add_node(t["id"], **t)
    for t in raw_terms:
        tid = t["id"]
        for parent_ref in t.get("is_a", []):
            parent_id = parent_ref.split(" ! ")[0].strip()
            if parent_id in G:
                G.add_edge(tid, parent_id, rel="is_a")
        for rel_ref in t.get("relations", []):
            rel_id = rel_ref.split(" ! ")[0].strip()
            if rel_id in G:
                G.add_edge(tid, rel_id, rel="has_relation")
                G.add_edge(rel_id, tid, rel="has_relation")
        for cat_ref in t.get("categories", []):
            cat_id = cat_ref.split(" ! ")[0].strip()
            if cat_id in G:
                G.add_edge(tid, cat_id, rel="has_metric_category")
        for vc_ref in t.get("value_concepts", []):
            vc_id = vc_ref.split(" ! ")[0].strip()
            if vc_id in G:
                G.add_edge(tid, vc_id, rel="has_value_concept")
    return G


def get_graph_neighbors(G, node_ids, hops=1):
    visited = set(node_ids)
    frontier = set(node_ids)
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            if n in G:
                next_frontier |= set(G.successors(n))
                next_frontier |= set(G.predecessors(n))
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier
    return visited

# Embeddings 

def _term_embedding_text(term):
    parts = [term.get("name", "")]
    if term.get("def"):
        parts.append(term["def"])
    if term.get("comment"):
        parts.append(term["comment"])
    for syn in term.get("synonyms", []):
        parts.append(syn)
    cats = [c.split(" ! ")[-1] for c in term.get("categories", [])]
    if cats:
        parts.append("Categories: " + ", ".join(cats))
    return ". ".join(parts)


def _obo_content_hash(raw_terms):
    content = json.dumps(
        [(t["id"], t["name"], t.get("def", "")) for t in raw_terms],
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _load_embedding_cache(obo_hash):
    meta_path = os.path.join(EMBEDDING_CACHE_DIR, "meta.json")
    matrix_path = os.path.join(EMBEDDING_CACHE_DIR, "embeddings.npz")
    if not (os.path.exists(meta_path) and os.path.exists(matrix_path)):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("cache_version") != EMBEDDING_CACHE_VERSION:
            logger.warning(
                "Embedding cache version mismatch (found %s, need %s); rebuilding.",
                meta.get("cache_version"), EMBEDDING_CACHE_VERSION,
            )
            return None
        if meta.get("obo_hash") != obo_hash:
            return None
        data = np.load(matrix_path)
        matrix = data["embeddings"]
        if not np.all(np.isfinite(matrix)):
            logger.warning(
                "Cached embeddings contain non-finite values; discarding "
                "cache and rebuilding."
            )
            return None
        return meta["term_ids"], matrix
    except Exception as e:
        logger.warning("Failed to load embedding cache: %s", e)
        return None


def _save_embedding_cache(obo_hash, term_ids, embedding_matrix):
    try:
        os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
        meta_path = os.path.join(EMBEDDING_CACHE_DIR, "meta.json")
        matrix_path = os.path.join(EMBEDDING_CACHE_DIR, "embeddings.npz")
        with open(meta_path, "w") as f:
            json.dump({
                "obo_hash": obo_hash,
                "term_ids": term_ids,
                "cache_version": EMBEDDING_CACHE_VERSION,
            }, f)
        np.savez_compressed(matrix_path, embeddings=embedding_matrix)
    except Exception as e:
        logger.warning("Failed to save embedding cache: %s", e)


def _sanitize_embedding_matrix(matrix):
    matrix = np.array(matrix, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    tiny = np.finfo(np.float64).tiny  
    matrix[np.abs(matrix) < tiny] = 0.0
    matrix = np.clip(matrix, -1e6, 1e6)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    bad_rows = ~np.all(np.isfinite(matrix), axis=1)
    if np.any(bad_rows):
        logger.warning(
            "Zeroing %d rows with non-finite values after normalization.",
            int(np.sum(bad_rows)),
        )
        matrix[bad_rows] = 0.0
    return matrix.astype(np.float32)


def build_or_load_embeddings(raw_terms, api_key):
    obo_hash = _obo_content_hash(raw_terms)
    cached = _load_embedding_cache(obo_hash)
    if cached is not None:
        term_ids, matrix = cached
        return term_ids, _sanitize_embedding_matrix(matrix)

    client = OpenAI(api_key=api_key, timeout=API_TIMEOUT)
    texts, term_ids = [], []
    for t in raw_terms:
        term_ids.append(t["id"])
        texts.append(_term_embedding_text(t))
    all_embeddings = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in resp.data:
            all_embeddings.append(item.embedding)
    embedding_matrix = np.array(all_embeddings, dtype=np.float32)
    embedding_matrix = _sanitize_embedding_matrix(embedding_matrix)

    _save_embedding_cache(obo_hash, term_ids, embedding_matrix)
    return term_ids, embedding_matrix


def embed_query(query_text, api_key):
    client = OpenAI(api_key=api_key, timeout=API_TIMEOUT)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text])
    vec = np.array(resp.data[0].embedding, dtype=np.float64)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vec = np.clip(vec, -1e6, 1e6)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.reshape(1, -1).astype(np.float32)


def retrieve_relevant_terms(
    query_name, query_desc, raw_terms, G, term_ids,
    embedding_matrix, api_key, top_k=TOP_K_EMBED,
    hops=GRAPH_HOP_LIMIT, max_total=MAX_CANDIDATES,
):
    query_text = f"{query_name}. {query_desc}"
    q_emb = embed_query(query_text, api_key)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = (
            q_emb.astype(np.float64) @ embedding_matrix.astype(np.float64).T
        ).flatten()
    sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)

    top_indices = np.argsort(sims)[::-1][:top_k]
    hit_ids = {term_ids[i] for i in top_indices}
    expanded_ids = get_graph_neighbors(G, hit_ids, hops=hops)
    all_ids = hit_ids | expanded_ids
    id_to_sim = {term_ids[i]: sims[i] for i in range(len(term_ids))}
    sorted_ids = sorted(all_ids, key=lambda x: id_to_sim.get(x, 0), reverse=True)
    id_to_term = {t["id"]: t for t in raw_terms}
    return [id_to_term[tid] for tid in sorted_ids[:max_total] if tid in id_to_term]

# Term helpers

def derive_value_type_from_is_a(term):
    for parent in term.get("is_a", []):
        for vname, vid in VTYPE_NAME_TO_ID.items():
            if parent.startswith(vid):
                return vname
    return None


def extract_name(ref_str):
    parts = ref_str.split(" ! ", 1)
    return parts[1] if len(parts) == 2 else parts[0]


def extract_id(ref_str):
    return ref_str.split(" ! ", 1)[0].strip()


def extract_id_name(ref_str):
    parts = ref_str.split(" ! ", 1)
    return {
        "id": parts[0].strip(),
        "name": parts[1].strip() if len(parts) == 2 else parts[0].strip(),
    }


def term_to_display_dict(term):
    vtype = derive_value_type_from_is_a(term)
    cats = [extract_name(c) for c in term.get("categories", [])]
    units = [extract_name(u) for u in term.get("units", [])]
    concepts = [extract_name(v) for v in term.get("value_concepts", [])]
    relations = [extract_id_name(r) for r in term.get("relations", [])]
    columns = [extract_name(c) for c in term.get("columns", [])]
    opt_columns = [extract_name(c) for c in term.get("optional_columns", [])]
    d = {"id": term["id"], "name": term["name"], "def": term["def"]}
    if term.get("comment"):
        d["comment"] = term["comment"]
    if term.get("synonyms"):
        d["synonyms"] = term["synonyms"]
    if vtype:
        d["value_type"] = vtype
    if cats:
        d["categories"] = cats
    if units:
        d["units"] = units
    if concepts:
        d["value_concepts"] = concepts
    if relations:
        d["relations"] = relations
    if columns:
        d["columns"] = columns
    if opt_columns:
        d["optional_columns"] = opt_columns
    if term.get("order"):
        d["order"] = extract_name(term["order"])
    if term.get("domain"):
        d["domain"] = extract_name(term["domain"])
    if term.get("is_obsolete"):
        d["is_obsolete"] = True
    if term.get("replaced_by"):
        d["replaced_by"] = term["replaced_by"]
    return d


def max_overlap_level(overlap_results):
    best, best_rank = None, 0
    for r in overlap_results:
        lvl = r.get("overlap_level", "low").lower()
        rank = OVERLAP_RANK.get(lvl, 0)
        if rank > best_rank:
            best_rank, best = rank, lvl
    return best


def find_raw_term_by_id(raw_terms, term_id):
    for t in raw_terms:
        if t["id"] == term_id:
            return t
    return None

# OBO block generation

def reconstruct_obo_block(raw_term):
    lines = ["[Term]", f"id: {raw_term['id']}", f"name: {raw_term['name']}"]
    if raw_term.get("def"):
        xrefs = ", ".join(raw_term.get("def_xrefs", ["PSI:MS"]))
        lines.append(f'def: "{raw_term["def"]}" [{xrefs}]')
    if raw_term.get("comment"):
        lines.append(f"comment: {raw_term['comment']}")
    for syn in raw_term.get("synonyms", []):
        lines.append(f'synonym: "{syn}" RELATED []')
    for parent in raw_term.get("is_a", []):
        lines.append(f"is_a: {parent}")
    if raw_term.get("xsd_value_type"):
        lines.append(f"relationship: has_value_type {raw_term['xsd_value_type']}")
    for u in raw_term.get("units", []):
        lines.append(f"relationship: has_units {u}")
    for vc in raw_term.get("value_concepts", []):
        lines.append(f"relationship: has_value_concept {vc}")
    for cat in raw_term.get("categories", []):
        lines.append(f"relationship: has_metric_category {cat}")
    for col in raw_term.get("columns", []):
        lines.append(f"relationship: has_column {col}")
    for col in raw_term.get("optional_columns", []):
        lines.append(f"relationship: has_optional_column {col}")
    for rel in raw_term.get("relations", []):
        lines.append(f"relationship: has_relation {rel}")
    if raw_term.get("order"):
        lines.append(f"relationship: has_order {raw_term['order']}")
    if raw_term.get("domain"):
        lines.append(f"relationship: has_domain {raw_term['domain']}")
    if raw_term.get("is_obsolete"):
        lines.append("is_obsolete: true")
    if raw_term.get("replaced_by"):
        lines.append(f"replaced_by: {raw_term['replaced_by']}")
    return "\n".join(lines)


def generate_obo_block(result):
    vtype = result.get("metric_value_type", "single value")
    vtype_id = VTYPE_NAME_TO_ID.get(vtype, "MS:4000003")
    categories = result.get("suggested_categories", [])
    units = result.get("suggested_units", [])
    xsd_type = result.get("suggested_xsd_type", "")
    relations = result.get("suggested_relations", [])
    columns = result.get("suggested_columns", [])
    obo_lines = [
        "[Term]",
        "id: MS:4000XXX",
        f"name: {result['suggested_name']}",
        f'def: "{result["suggested_def"]}" [PSI:MS]',
        f"is_a: {vtype_id} ! {vtype}",
    ]
    if xsd_type:
        obo_lines.append(f"relationship: has_value_type {xsd_type} ! The allowed value-type for this CV term")
    for unit in units:
        uid = UNIT_NAME_TO_ID.get(unit)
        if uid:
            obo_lines.append(f"relationship: has_units {uid} ! {unit}")
        else:
            logger.warning("Unknown unit '%s' — skipping from OBO block", unit)
    for cat_name in categories:
        cat_id = CATEGORY_NAME_TO_ID.get(cat_name)
        if cat_id:
            obo_lines.append(f"relationship: has_metric_category {cat_id} ! {cat_name}")
    for rel in relations:
        if rel.get("id"):
            obo_lines.append(f"relationship: has_relation {rel['id']} ! {rel.get('name', '')}")
    for col in columns:
        obo_lines.append(f"relationship: has_column ... ! {col}")
    return "\n".join(obo_lines)


# Accepted metrics persistence

def load_accepted_metrics():
    if os.path.exists(NEW_METRICS_FILE):
        with open(NEW_METRICS_FILE, "r") as f:
            return json.load(f)
    return []


def save_accepted_metrics(accepted):
    with open(NEW_METRICS_FILE, "w") as f:
        json.dump(accepted, f, indent=2)


def append_new_metric(result, proposed_name, proposed_desc):
    accepted = load_accepted_metrics()
    existing_names = {m.get("suggested_name", "").lower() for m in accepted}
    if result.get("suggested_name", "").lower() in existing_names:
        return False, "already_saved"
    record = {
        "proposed_name": proposed_name,
        "proposed_description": proposed_desc,
        "suggested_name": result.get("suggested_name"),
        "suggested_def": result.get("suggested_def"),
    }
    for dim_key in CLASSIFICATION_SCHEMA:
        record[dim_key] = result.get(dim_key)
    record["suggested_categories"] = result.get("suggested_categories")
    record["suggested_units"] = result.get("suggested_units")
    record["suggested_xsd_type"] = result.get("suggested_xsd_type")
    record["suggested_relations"] = result.get("suggested_relations")
    record["suggested_columns"] = result.get("suggested_columns")
    record["max_overlap_level"] = max_overlap_level(result.get("overlap_results", []))
    record["overlap_results"] = result.get("overlap_results", [])
    record["verdict_summary"] = result.get("verdict_summary")
    record["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    accepted.append(record)
    save_accepted_metrics(accepted)
    return True, "saved"


# Prompt building

def build_classification_prompt_section():
    lines = [
        "Each QC metric must be classified along seven independent "
        "dimensions. For each dimension, choose exactly one value.",
        "",
    ]
    for i, (dim_key, dim_info) in enumerate(CLASSIFICATION_SCHEMA.items(), 1):
        lines.append(f"  Dimension {i} -- {dim_info['label']} (JSON key: \"{dim_key}\"):")
        lines.append(f"  {dim_info['description']}")
        for val_name, val_desc in dim_info["values"].items():
            lines.append(f'    - "{val_name}": {val_desc}')
        lines.append("")
    lines += [
        "CRITICAL DISAMBIGUATION RULES:",
        "",
        "  workflow_stage vs data_dependency -- these are INDEPENDENT:",
        "    workflow_stage = where the MEASURED QUANTITY physically originates in the pipeline.",
        "    data_dependency = what INPUT DATA is needed to COMPUTE the metric.",
        "    Example: 'precursor ppm deviation mean' measures MS1 mass accuracy "
        "(workflow_stage = MS1 acquisition stage) but requires identification "
        "results as a filter (data_dependency = identification results).",
        "    Use 'identification stage' ONLY when the quantity itself is an "
        "identification quality measure (FDR, ID rate, PSM count).",
        "    Use 'quantification stage' ONLY when the quantity itself is a "
        "quantitative accuracy/precision measure.",
        "",
        "  quality_directionality -- decision rules:",
        '    1. Purity, coverage, rate, completeness fraction where 1.0 = perfect -> "higher is better".',
        '    2. Error, deviation, contamination, bad event count -> "lower is better".',
        '    3. SIGNED deviation centered on a target (mass deviation where 0 ppm is ideal, temperature) -> "target range".',
        '    4. "context dependent" ONLY when direction genuinely varies by experimental design.',
        "",
    ]
    return "\n".join(lines)


def build_system_prompt(candidate_display_terms, total_count, active_count):
    metrics_text = json.dumps(candidate_display_terms, indent=1)
    classification_section = build_classification_prompt_section()
    legacy_cats = ", ".join(f'"{c}"' for c in CATEGORY_NAME_TO_ID.keys())
    dim_keys_example = ",\n".join(f'  "{k}": "..."' for k in CLASSIFICATION_SCHEMA)

    json_example = (
        "{\n"
        '  "needs_more_detail": false,\n'
        '  "clarification_question": null,\n'
        '  "suggested_name": "metric name",\n'
        '  "suggested_def": "OBO definition text.",\n'
        "\n"
        + dim_keys_example + ",\n"
        "\n"
        '  "suggested_categories": ["ID free metric", "MS2 metric"],\n'
        '  "suggested_units": ["parts per million"],\n'
        '  "suggested_xsd_type": "xsd:float",\n'
        '  "suggested_relations": [\n'
        '    {"id": "MS:4000072", "name": "observed mass accuracy"}\n'
        "  ],\n"
        '  "suggested_columns": [],\n'
        "\n"
        '  "overlap_results": [\n'
        "    {\n"
        '      "id": "MS:4000026",\n'
        '      "name": "fragment ppm deviation median",\n'
        '      "overlap_level": "moderate",\n'
        '      "is_obsolete": false,\n'
        '      "reasoning": "Both measure mass accuracy deviation but at different MS levels."\n'
        "    }\n"
        "  ],\n"
        '  "is_new": true,\n'
        '  "verdict_summary": "This metric is new."\n'
        "}"
    )

    prompt_parts = [
        "You are an expert in mass spectrometry quality control metrics, "
        "specifically the PSI-MS controlled vocabulary used in mzQC files.\n",
        "Below is a CURATED SUBSET of the most relevant existing QC terms, "
        f"retrieved from the official OBO ontology ({total_count} terms total, "
        f"{active_count} active). These are the terms most likely to overlap "
        "with the proposed metric. Each entry has:\n"
        "  - id, name, def, comment, synonyms\n"
        '  - value_type: "single value", "n-tuple"/"tuple", "table", "matrix"\n'
        "  - categories: legacy OBO metric categories\n"
        "  - units: measurement units\n"
        "  - value_concepts: statistical concepts\n"
        "  - relations: list of related CV terms via has_relation\n"
        "  - columns / optional_columns: for table-type metrics\n"
        "  - order: ordering hint\n"
        "  - domain: value domain constraint\n"
        "  - is_obsolete, replaced_by\n",
        "<relevant_qc_metrics>\n" + metrics_text + "\n</relevant_qc_metrics>\n",
        f"NOTE: {len(candidate_display_terms)} of {total_count} terms shown "
        "(pre-filtered by semantic similarity and graph neighbors). If you "
        "believe a relevant term might exist outside this subset, mention it "
        "in your verdict_summary.\n",
        "A user is proposing a new QC metric. Perform the following steps.\n",
        "STEP 1 -- ASSESS DESCRIPTION QUALITY:\n"
        "It must specify: (a) what quantity is measured, (b) how it is computed, "
        "(c) what MS level, (d) ID-based or ID-free, (e) unit/value type.\n"
        "If any are missing, set needs_more_detail to true and write a specific "
        "clarification_question. Still provide best attempt at all other fields.\n",
        "STEP 2 -- GENERATE FORMAL NAME:\n"
        "Short, precise, lowercase (except proper nouns). Follow patterns: "
        '"X distribution mean", "number of X", "X quantiles", etc. No MS:4000XXX id.\n',
        "STEP 3 -- GENERATE FORMAL DEFINITION:\n"
        'OBO-style def: "..." text. Precise, technical, self-contained. '
        "Reference other MS terms by accession where relevant.\n",
        "STEP 4 -- SEVEN-DIMENSION CLASSIFICATION:\n" + classification_section,
        "STEP 5 -- LEGACY OBO FIELDS:\n"
        f"  - suggested_categories: list from: {legacy_cats}.\n"
        "  - suggested_units: list of unit strings.\n"
        '  - suggested_xsd_type: "xsd:float" or "xsd:int".\n',
        "STEP 6 -- RELATIONSHIPS (has_relation):\n"
        "Identify any existing CV terms that are semantically related to the "
        "proposed metric via has_relation. These are terms that:\n"
        "  - Define the base quantity being summarized.\n"
        "  - Define a parent concept the metric is derived from.\n"
        "  - Are companion metrics computed from the same data.\n"
        "  - Are referenced in the definition by accession.\n"
        'Report as suggested_relations: a list of objects with "id" and "name".\n'
        "Only include terms where there is a genuine semantic link. If no "
        "has_relation is appropriate, use an empty list.\n",
        'STEP 7 -- TABLE COLUMNS (if metric_value_type is "table"):\n'
        "If the metric is a table, specify suggested_columns. Otherwise omit "
        "or use an empty list.\n",
        "STEP 8 -- OVERLAP ANALYSIS (BE THOROUGH AND CONSISTENT):\n"
        "Compare against every term in the provided subset by semantic meaning. "
        "Use all fields including relations.\n\n"
        "OVERLAP LEVEL DEFINITIONS:\n"
        '  "duplicate"  = SAME quantity, SAME statistic, SAME scope, SAME filtering, SAME value type.\n'
        '  "high"       = same quantity but output is derivable from or contained in an existing term.\n'
        '  "moderate"   = related quantity at the same MS level (mean vs sigma, precursor vs fragment).\n'
        '  "low"        = loosely related concept.\n\n'
        "CONSISTENCY RULES:\n"
        '  - mean vs sigma of same distribution = "moderate" (different properties).\n'
        '  - single-value statistic already contained in an existing distribution/quantile term = "high".\n'
        "  - Sharing a unit or category alone is NOT enough for overlap.\n\n"
        "For each overlapping term report: id, name, overlap_level, is_obsolete, reasoning.\n",
        "STEP 9 -- VERDICT:\n"
        '  - Any "duplicate": metric IS a duplicate.\n'
        '  - Highest "high": flag HIGH OVERLAP.\n'
        "  - Otherwise: metric is NEW.\n",
        "RESPONSE FORMAT -- ONLY valid JSON, no markdown fences:\n\n"
        + json_example + "\n\n"
        "Use the EXACT string values for each dimension. Do not invent new values.",
    ]

    return "\n".join(prompt_parts)

# LLM call

def call_gpt(messages, api_key):
    client = OpenAI(api_key=api_key, timeout=API_TIMEOUT)
    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=messages,
            max_completion_tokens=16000,
        )
    except APITimeoutError:
        return None, "The API request timed out. Please try again."
    except APIError as e:
        return None, f"OpenAI API error: {e}"

    raw = response.choices[0].message.content.strip()
    return _parse_json_response(raw)


def _parse_json_response(raw):
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start : i + 1]), None
                except json.JSONDecodeError:
                    pass
    return None, (
        "Could not parse the model response as JSON.\n\n"
        f"Raw response (first 1500 chars):\n{raw[:1500]}"
    )

# Streamlit 

OVERLAP_DISPLAY_LIMIT = 10


def render_overlap_label(level):
    st.write(f"Overlap: **{level.upper()}**")


def render_classification_card(result):
    st.subheader("7-Dimension Classification")
    for dim_key, dim_info in CLASSIFICATION_SCHEMA.items():
        val = result.get(dim_key, "") or "not determined"
        st.write(f"**{dim_info['label']}:** {val}")


def render_sidebar(display_terms, active_count, api_key):
    st.sidebar.header("Configuration")

    if api_key:
        st.sidebar.success("API key loaded.")
    else:
        st.sidebar.error(
            "No API key found. Set OPENAI_API_KEY env var or .streamlit/secrets.toml."
        )

    st.sidebar.markdown("---")
    st.sidebar.write("Source: qc_metrics_specific.obo")
    st.sidebar.write(f"Number of Terms: {len(display_terms)} ")
    st.sidebar.write("Classification: 7 dimensions")
    st.sidebar.write("Model: GPT-5.4 Mini")


def render_accepted_sidebar():
    accepted = load_accepted_metrics()
    st.sidebar.markdown("---")
    st.sidebar.header(f"Accepted New Metrics ({len(accepted)})")
    if accepted:
        for i, m in enumerate(accepted, 1):
            with st.sidebar.expander(f"{i}. {m.get('suggested_name', 'unnamed')}"):
                ad = m.get("analytical_dimension")
                if ad:
                    st.write(f"**Analytical dim:** {ad}")
                vt = m.get("metric_value_type", "N/A")
                st.write(f"**Value type:** {vt}")
                lvl = m.get("max_overlap_level") or "none"
                st.write(f"**Max overlap:** {lvl.upper()}")
                rels = m.get("suggested_relations", [])
                if rels:
                    rel_strs = [f"{r['id']} ({r['name']})" for r in rels]
                    st.write(f"**Relations:** {', '.join(rel_strs)}")
                st.caption(f"Saved: {m.get('saved_at', 'unknown')}")
                st.code(m.get("suggested_def", ""), language=None)
        st.sidebar.download_button(
            label="Download accepted_new_metrics.json",
            data=json.dumps(accepted, indent=2),
            file_name="accepted_new_metrics.json",
            mime="application/json",
        )
    else:
        st.sidebar.caption("No new metrics accepted yet.")


def render_vocabulary_browser(display_terms):
    with st.expander(f"Browse existing QC vocabulary ({len(display_terms)} terms)", expanded=False):
        for t in display_terms:
            cats = t.get("categories", [])
            obs = " [OBSOLETE]" if t.get("is_obsolete") else ""
            cat_str = ", ".join(cats) if cats else ""
            rels = t.get("relations", [])
            rel_str = ""
            if rels:
                rel_str = "  ->  " + ", ".join(
                    f"{r['id']} ({r['name']})" for r in rels
                )
            label = f"{t['id']}  |  {t['name']}{obs}"
            if cat_str:
                label += f"  |  [{cat_str}]"
            label += rel_str
            st.text(label)


def render_results(result, raw_terms):
    st.markdown("---")

    top_level = max_overlap_level(result.get("overlap_results", []))
    is_duplicate = top_level == "duplicate"
    is_high = top_level == "high"
    is_new_metric = (
        not result.get("needs_more_detail")
        and result.get("is_new")
        and top_level not in ("duplicate", "high")
    )

    _render_verdict(result, raw_terms, is_duplicate, is_high, is_new_metric)

    st.write(result.get("verdict_summary", ""))

    if is_new_metric and result.get("suggested_name"):
        saved, status = append_new_metric(
            result,
            st.session_state.get("proposed_name", ""),
            st.session_state.get("proposed_desc", ""),
        )
        if saved:
            st.write(
                "This metric has been saved to **accepted_new_metrics.json**. "
                "See the sidebar for all accepted metrics."
            )
        elif status == "already_saved":
            st.write("This metric was already saved previously.")

    has_dims = any(result.get(k) for k in CLASSIFICATION_SCHEMA)
    if has_dims:
        st.markdown("---")
        render_classification_card(result)

    if is_new_metric and result.get("suggested_name"):
        st.markdown("---")
        st.subheader("Generated OBO Entry")
        obo_block = generate_obo_block(result)
        st.code(obo_block, language=None)

    _render_overlap_table(result, is_new_metric)

    return is_new_metric


def _render_verdict(result, raw_terms, is_duplicate, is_high, is_new_metric):
    if result.get("needs_more_detail"):
        st.write("**VERDICT:** More detail is needed to fully evaluate this proposal.")
    elif is_duplicate:
        st.write("**VERDICT:** DUPLICATE -- this metric already exists in the vocabulary.")
        for item in result.get("overlap_results", []):
            if item.get("overlap_level", "").lower() == "duplicate":
                match_id = item.get("id", "")
                st.info(f"Use existing term **{match_id}** ({item.get('name')}) instead.")
                match_term = find_raw_term_by_id(raw_terms, match_id)
                if match_term:
                    st.subheader("Existing Term OBO Entry")
                    st.code(reconstruct_obo_block(match_term), language=None)
                break
    elif is_high:
        st.write("**VERDICT:** HIGH OVERLAP -- this may be redundant with an existing term.")
        for item in result.get("overlap_results", []):
            if item.get("overlap_level", "").lower() == "high":
                match_id = item.get("id", "")
                st.warning(
                    f"Closely related to **{match_id}** ({item.get('name')}). "
                    "Consider whether a new term is needed."
                )
                match_term = find_raw_term_by_id(raw_terms, match_id)
                if match_term:
                    with st.expander(f"View existing term {match_id}", expanded=False):
                        st.code(reconstruct_obo_block(match_term), language=None)
                break
    elif is_new_metric:
        st.write("**VERDICT:** NEW METRIC -- no duplicate found.")
    else:
        st.write("**VERDICT:** Likely a new metric.")


def _render_overlap_table(result, is_new_metric):
    overlaps = result.get("overlap_results", [])
    if overlaps:
        st.markdown("---")
        display_limit = OVERLAP_DISPLAY_LIMIT
        shown_count = min(len(overlaps), display_limit)
        if len(overlaps) > display_limit:
            st.subheader(
                f"Overlap Analysis -- top {shown_count} of "
                f"{len(overlaps)} related term(s)"
            )
        else:
            st.subheader(
                f"Overlap Analysis -- {len(overlaps)} related term(s)"
            )
        level_order = ["duplicate", "high", "moderate", "low"]
        sorted_overlaps = sorted(
            overlaps,
            key=lambda x: (
                level_order.index(x.get("overlap_level", "low").lower())
                if x.get("overlap_level", "low").lower() in level_order
                else 99
            ),
        )
        for item in sorted_overlaps[:display_limit]:
            lvl = item.get("overlap_level", "low").lower()
            obs_tag = " (OBSOLETE)" if item.get("is_obsolete") else ""
            st.write(f"**{item.get('id', '?')}** -- {item.get('name', '?')}{obs_tag}")
            render_overlap_label(lvl)
            st.caption(item.get("reasoning", ""))
            st.write("")
    elif is_new_metric and not result.get("needs_more_detail"):
        st.markdown("---")
        st.write(
            "This metric does not overlap with any existing term in the PSI-MS vocabulary."
        )


def render_clarification_ui(result, api_key):
    if not (result.get("needs_more_detail") and result.get("clarification_question")):
        return False

    st.markdown("---")
    st.subheader("Clarification Needed")
    st.write(result["clarification_question"])
    follow_up = st.text_area("Provide additional detail:", key="followup_text", height=100)

    if st.button("Re-Analyze"):
        if not follow_up.strip():
            st.warning("Please type some additional detail first.")
            return False

        st.session_state.messages.append({
            "role": "user",
            "content": (
                f'Additional information:\n\n"{follow_up}"\n\n'
                "Please re-evaluate the proposed metric."
            ),
        })
        with st.spinner("Re-analyzing with GPT-5.4 Mini ..."):
            result2, error2 = call_gpt(st.session_state.messages, api_key)
            if error2:
                st.error(error2)
                return False
            st.session_state.result = result2
            st.session_state.messages.append({
                "role": "assistant",
                "content": json.dumps(result2),
            })
            st.rerun()
    return False


@st.cache_data
def load_metrics():
    obo_path = os.path.join(SCRIPT_DIR, "qc_metrics_specific.obo")
    if not os.path.exists(obo_path):
        st.error(
            f"OBO file not found at {obo_path}. "
            "Place qc_metrics_specific.obo next to this script."
        )
        return [], []
    raw_terms = parse_obo(obo_path)
    display_terms = [term_to_display_dict(t) for t in raw_terms]
    return raw_terms, display_terms


@st.cache_resource
def load_knowledge_graph(_raw_terms):
    return build_knowledge_graph(_raw_terms)


@st.cache_resource
def load_embeddings(_raw_terms, api_key):
    return build_or_load_embeddings(_raw_terms, api_key)


def _resolve_api_key():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            api_key = ""
    return api_key


def main():
    st.set_page_config(page_title="QC Metric Proposal Analyzer", layout="wide")

    raw_terms, display_terms = load_metrics()
    if not raw_terms:
        return

    active_count = sum(1 for t in display_terms if not t.get("is_obsolete"))
    api_key = _resolve_api_key()

    st.title("PSI-MS QC Metric Proposal Analyzer")
    st.write(
        f"Validates new metric proposals against "
        f"**{len(display_terms)}** existing QC terms "
        f"({active_count} active, {len(display_terms) - active_count} obsolete) "
        f"parsed from the official OBO ontology using **GPT-5.4 Mini**."
    )

    render_sidebar(display_terms, active_count, api_key)
    render_accepted_sidebar()

    for key, default in [
        ("messages", []), ("result", None),
        ("proposed_name", ""), ("proposed_desc", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    G, term_ids, embedding_matrix = None, None, None
    if api_key:
        G = load_knowledge_graph(raw_terms)
        term_ids, embedding_matrix = load_embeddings(raw_terms, api_key)

    st.subheader("Propose a New QC Metric")
    proposed_name = st.text_input(
        "Proposed metric name", placeholder="e.g. MS2 spectral entropy median"
    )
    proposed_desc = st.text_area(
        "Description -- explain what this metric measures and how it is computed",
        placeholder=(
            "Include:\n"
            "- What quantity is being measured\n"
            "- How it is computed (mean, median, ratio, count, etc.)\n"
            "- Whether it requires identification results (ID-based) or not (ID-free)\n"
            "- What MS level (MS1, MS2, run-level)\n"
            "- What unit (ppm, seconds, fraction, count, etc.)"
        ),
        height=160,
    )

    can_submit = bool(proposed_name and proposed_desc and api_key)
    analyze_clicked = st.button("Analyze Proposal", disabled=not can_submit)

    if not api_key and (proposed_name or proposed_desc):
        st.warning(
            "No API key configured. Set OPENAI_API_KEY env var or "
            ".streamlit/secrets.toml."
        )

    if analyze_clicked:
        st.session_state.proposed_name = proposed_name
        st.session_state.proposed_desc = proposed_desc

        with st.spinner("Retrieving relevant terms ..."):
            candidate_raw = retrieve_relevant_terms(
                proposed_name, proposed_desc, raw_terms, G,
                term_ids, embedding_matrix, api_key,
                top_k=TOP_K_EMBED, hops=GRAPH_HOP_LIMIT, max_total=MAX_CANDIDATES,
            )
            candidate_display = [term_to_display_dict(t) for t in candidate_raw]

        system_prompt = build_system_prompt(
            candidate_display, len(display_terms), active_count
        )

        st.session_state.messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "I want to propose a new QC metric for the PSI-MS "
                    "controlled vocabulary.\n\n"
                    f'Proposed name: "{proposed_name}"\n\n'
                    f'Description: "{proposed_desc}"'
                ),
            },
        ]

        with st.spinner("Analyzing with GPT-5.4 Mini ..."):
            result, error = call_gpt(st.session_state.messages, api_key)
            if error:
                st.error(error)
                return
            st.session_state.result = result
            st.session_state.messages.append({
                "role": "assistant",
                "content": json.dumps(result),
            })

    result = st.session_state.result
    if result is None:
        render_vocabulary_browser(display_terms)
        return

    render_results(result, raw_terms)
    render_clarification_ui(result, api_key)


if __name__ == "__main__":
    main()
