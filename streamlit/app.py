import os
import json
import re
import datetime
import hashlib
import logging
import time
import urllib.request
import urllib.error

import numpy as np
import networkx as nx
import streamlit as st
from openai import OpenAI, BadRequestError, APIError, APITimeoutError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_METRICS_FILE = os.path.join(SCRIPT_DIR, "accepted_new_metrics.json")
EMBEDDING_CACHE_DIR = os.path.join(SCRIPT_DIR, ".kg_embedding_cache")
EMBEDDING_CACHE_VERSION = 2
EMBEDDING_MODEL = "text-embedding-3-small"

#retrieval budget
EMBED_QUOTA = 25
GRAPH_QUOTA = 15
GRAPH_HOP_LIMIT = 2
MAX_CANDIDATES = EMBED_QUOTA + GRAPH_QUOTA

HUB_DEGREE_LIMIT = 60

API_TIMEOUT = 120
MAX_REQUESTS_PER_SESSION = 20
LLM_MODEL = "gpt-5.4"
MAX_COMPLETION_TOKENS = 16000

#the CV comes from the psi-ms-cv
OBO_BLOB_URL = "https://github.com/HUPO-PSI/psi-ms-CV/blob/master/psi-ms.obo"
OBO_RAW_URL = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"
OBO_COMMITS_API = (
    "https://api.github.com/repos/HUPO-PSI/psi-ms-CV/commits"
    "?path=psi-ms.obo&per_page=1"
)
OBO_PINNED_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/{sha}/psi-ms.obo"
)
CV_NAME = "Proteomics Standards Initiative Mass Spectrometry Ontology"
CV_CACHE_DIR = os.path.join(SCRIPT_DIR, ".psi_ms_cv_cache")
CV_CACHE_TTL = 24 * 3600
CV_DOWNLOAD_TIMEOUT = 90
CV_API_TIMEOUT = 20
USER_AGENT = "psi-ms-qc-metric-proposal-analyzer/2.1"

LOCAL_OBO_FALLBACK = os.path.join(SCRIPT_DIR, "psi-ms.obo")

logger = logging.getLogger(__name__)

OVERLAP_RANK = {"duplicate": 4, "high": 3, "moderate": 2, "low": 1}


QC_ROOT_ID = "MS:4000000"
QC_METRIC_ID = "MS:4000001"
QC_VALUE_TYPE_ROOT_ID = "MS:4000002"
QC_CATEGORY_ROOT_ID = "MS:4000007"

QC_NON_METRIC_ID = "MS:4000080"
QC_NON_METRIC_NAME = "QC non-metric term"


QC_NAMESPACE_MIN = 4000000
QC_NAMESPACE_MAX = 4999999

#spec 4.1
MAX_NAME_LENGTH = 100
NAME_ALLOWED_RE = re.compile(r"^[A-Za-z0-9 \-_,\.]+$")

#placeholder accessions: the metric gets MS:4000XXX
NEW_TERM_PLACEHOLDER = "MS:4000XXX"
NEW_COLUMN_PLACEHOLDER_TEMPLATE = "MS:4000XX{n}"

ALLOWED_XSD_TYPES = [
    "xsd:float", "xsd:double", "xsd:int", "xsd:integer",
    "xsd:nonNegativeInteger", "xsd:positiveInteger",
    "xsd:string", "xsd:boolean", "xsd:dateTime", "xsd:anyURI",
]

QC_PREFERRED_XSD_TYPES = ["xsd:float", "xsd:int", "xsd:integer", "xsd:string"]


DEF_BOILERPLATE_RE = re.compile(
    r"^\s*(a\s+|the\s+)?(quality control|qc)\s+metric\s+"
    r"(that\s+|which\s+)?"
    r"(report(s|ing)?|describ(es|ing)|measur(es|ing)|"
    r"provid(es|ing)|giv(es|ing)|captur(es|ing)|quantif(ies|ying))\s+",
    re.IGNORECASE,
)

#standard abbreviations that must stay uppercase in metric names
MS_ABBREVIATIONS = {
    "ms1": "MS1", "ms2": "MS2", "ms3": "MS3", "msn": "MSn",
    "dda": "DDA", "dia": "DIA", "swath": "SWATH",
    "srm": "SRM", "prm": "PRM", "mrm": "MRM",
    "tic": "TIC", "xic": "XIC", "bpc": "BPC",
    "fwhm": "FWHM", "fdr": "FDR", "psm": "PSM",
    "ccs": "CCS", "ims": "IMS", "tof": "TOF",
    "esi": "ESI", "maldi": "MALDI", "desi": "DESI",
    "lc": "LC", "gc": "GC", "ce": "CE",
    "rt": "RT", "m/z": "m/z", "s/n": "S/N",
    "bqc": "BQC", "tqc": "TQC", "cv": "CV", "lod": "LOD", "loq": "LOQ",
}

CLASSIFICATION_SCHEMA = {
    "analytical_dimension": {
        "label": "Analytical dimension",
        "predicate": "is_a",
        "description": "What type of QC metric this is (the metric's place in the taxonomy).",
        "values": {
            "acquisition coverage metric": "How comprehensively data were collected (scan counts, sampling density, number of spectra, MS1 scan window m/z limits, configured acquisition ranges).",
            "mass accuracy metric": "Deviation between observed and theoretical m/z.",
            "intensity stability metric": "Variation of signal intensity over time.",
            "chromatographic performance metric": "Separation performance (peak width, symmetry, RT reproducibility).",
            "ionization quality metric": "Properties of the precursor ion population (charge-state distribution, adducts).",
            "ion mobility metric": "IMS resolution, drift-time/CCS accuracy and reproducibility.",
            "spectral quality metric": "Quality of individual spectra (peak density, S/N, completeness, entropy).",
            "fragmentation efficiency metric": "Effectiveness of precursor fragmentation to produce interpretable spectra.",
            "isolation purity metric": "MS2 precursor isolation selectivity, co-isolation of interfering species, or MS2 isolation window configuration (isolation window width, isolation target m/z, isolation boundaries). NOT for MS1 scan window limits, those are acquisition coverage.",
            "identification confidence metric": "Reliability of identifications (FDR, ID rate).",
            "quantification precision metric": "Reproducibility or variability of quantitative results (CV across replicates, response curve linearity, signal-to-blank ratio).",
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
            "instrument calibration stage": "The quantity is derived from calibration routines, response curves, or control samples.",
            "data preprocessing stage": "The quantity describes baseline correction, noise removal, or peak picking.",
            "identification stage": "The quantity is ITSELF an identification quality measure (FDR, ID rate, PSM count). Not for metrics that merely use IDs as a filter.",
            "quantification stage": "The quantity is ITSELF a quantitative accuracy or precision measure (CV of concentrations, ratio reproducibility).",
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
            "reference data": "Requires comparison to external standards or reference files (iRT peptides, calibration standards, procedural blanks, pooled QC samples).",
        },
    },
    "measurement_scope": {
        "label": "Measurement scope",
        "predicate": "has_measurement_scope",
        "description": (
            "At what aggregation level the metric summarizes data. "
            "TIE-BREAK FOR TABLES: the scope of a table is the scope of ONE "
            "ROW, not the scope of the input it was computed from. A table "
            "with one row per feature is feature level even when each row "
            "was computed across many runs (say so with the "
            "'multiple runs based metric' category instead). A table with "
            "one row per run is run level, one row per spectrum is spectrum "
            "level. Use batch level or study level ONLY when the metric "
            "reports a single aggregate for the whole batch or study."
        ),
        "values": {
            "spectrum level": "Per-spectrum metrics (one value per spectrum).",
            "pixel/voxel level": "Per-pixel metrics in imaging or spatial omics.",
            "feature level": "Per feature (peptide, compound, lipid, or chromatographic peak).",
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
            "targeted acquisition": "For SRM, PRM, MRM, or other targeted workflows.",
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
            "higher is better": "Increasing values always indicate improved quality (identification rate, purity fraction, coverage, signal-to-noise, signal-to-blank, coefficient of determination).",
            "lower is better": "Decreasing values always indicate improved quality (FDR, absolute mass error, noise level, contamination fraction, coefficient of variation, number of empty scans).",
            "context dependent": "Interpretation genuinely varies by experimental design or method, there is no single direction that is universally better (charge-state fractions, peak density, spectral entropy, regression slope).",
            "target range": "Optimal quality corresponds to a specific value or interval, with deviations in either direction being bad (signed mass deviation centered on 0 ppm, temperature, regression intercept centered on 0).",
            "categorical": "Quality expressed as discrete categories (pass/fail, OK/warning/error).",
            "trend": "Intended for temporal monitoring and drift detection rather than direct ranking (instrument drift over time, long-term TIC trend).",
        },
    },
    "metric_value_type": {
        "label": "Metric value type",
        "predicate": "has_value_type",
        "description": "The structural format of the metric's reported value(s) in mzQC (see mzQC spec section 7).",
        "values": {
            "single value": "A single numeric, string, or boolean value. has_units is REQUIRED.",
            "n-tuple": "An ordered JSON array of values of the same type AND the same unit (quantiles, min/max pairs, quartile fractions). has_units is REQUIRED and uniform.",
            "table": "A JSON object of named columns, all of equal length; each column carries its own unit through its own CV term. has_column is REQUIRED, has_units MUST NOT be given on the table itself.",
            "matrix": "A JSON array of arrays of uniform length and uniform type. has_units is REQUIRED and uniform.",
        },
    },
}

#fallbacks
FALLBACK_VTYPE_NAME_TO_ID = {
    "single value": "MS:4000003",
    "n-tuple": "MS:4000004",
    "table": "MS:4000005",
    "matrix": "MS:4000006",
}

#documentation discrepancy -> section 7 of the mzQC document
VTYPE_ALIASES = {
    "tuple": "n-tuple",
    "ntuple": "n-tuple",
    "n tuple": "n-tuple",
    "array": "n-tuple",
    "list": "n-tuple",
}

FALLBACK_CATEGORY_NAME_TO_ID = {
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
}

UNIT_ALIAS_TO_ID = {
    "parts per million": "UO:0000169", "ppm": "UO:0000169",
    "count unit": "UO:0000189", "count": "UO:0000189", "counts": "UO:0000189",
    "second": "UO:0000010", "seconds": "UO:0000010", "s": "UO:0000010",
    "fraction": "UO:0000191",
    "dalton": "UO:0000221", "da": "UO:0000221",
    "percent": "UO:0000187", "%": "UO:0000187",
    "intensity unit": "MS:1000043",
    "pressure unit": "UO:0000109",
    "hertz": "UO:0000106", "hz": "UO:0000106",
    "electronvolt": "UO:0000266", "ev": "UO:0000266",
    "millisecond": "UO:0000028", "ms": "UO:0000028",
    "minute": "UO:0000031", "min": "UO:0000031",
    "m/z": "MS:1000040", "thomson": "MS:1000040", "th": "MS:1000040",
    "number of detector counts": "MS:1000131",
    "counts per second": "MS:1000814", "cps": "MS:1000814",
    "volt": "UO:0000218", "v": "UO:0000218",
    "degree celsius": "UO:0000027", "celsius": "UO:0000027",
    "pascal": "UO:0000110", "pa": "UO:0000110",
    "kelvin": "UO:0000012",
    "nanometer": "UO:0000018", "micrometer": "UO:0000017",
    "millimeter": "UO:0000016", "meter": "UO:0000008",
    "absorbance unit": "UO:0000269",
    "mass unit": "UO:0000002",
    "kilodalton": "UO:0000222", "kda": "UO:0000222",
    "ratio": "UO:0000190",
    "milliliter": "UO:0000098", "microliter": "UO:0000101",
    "square angstrom": "UO:0000324",
    "volt-second per square centimeter": "MS:1002814",
    "percent of base peak": "MS:1000132",
    "dimensionless unit": "UO:0000186", "dimensionless": "UO:0000186",
    "unitless": "UO:0000186", "none": "UO:0000186", "no unit": "UO:0000186",
    "arbitrary unit": "UO:0000186", "arbitrary units": "UO:0000186",
    "a.u.": "UO:0000186", "au": "UO:0000186",
    "nat": "UO:0000186", "nats": "UO:0000186",
    "bit": "UO:0000232", "bits": "UO:0000232", "shannon": "UO:0000232",
}

#canonical name for every accession
EXTRA_UNIT_NAMES = {
    "UO:0000002": "mass unit",
    "UO:0000008": "meter",
    "UO:0000010": "second",
    "UO:0000012": "kelvin",
    "UO:0000016": "millimeter",
    "UO:0000017": "micrometer",
    "UO:0000018": "nanometer",
    "UO:0000027": "degree Celsius",
    "UO:0000028": "millisecond",
    "UO:0000031": "minute",
    "UO:0000098": "milliliter",
    "UO:0000101": "microliter",
    "UO:0000106": "hertz",
    "UO:0000109": "pressure unit",
    "UO:0000110": "pascal",
    "UO:0000169": "parts per million",
    "UO:0000186": "dimensionless unit",
    "UO:0000187": "percent",
    "UO:0000189": "count unit",
    "UO:0000190": "ratio",
    "UO:0000191": "fraction",
    "UO:0000218": "volt",
    "UO:0000221": "dalton",
    "UO:0000222": "kilodalton",
    "UO:0000232": "bit",
    "UO:0000266": "electronvolt",
    "UO:0000269": "absorbance unit",
    "UO:0000324": "square angstrom",
    "MS:1000040": "m/z",
    "MS:1000043": "intensity unit",
    "MS:1000131": "number of detector counts",
    "MS:1000132": "percent of base peak",
    "MS:1000814": "counts per second",
    "MS:1002814": "volt-second per square centimeter",
}


#rate limiting

def check_rate_limit():

    if "request_count" not in st.session_state:
        st.session_state.request_count = 0
        st.session_state.first_request_time = time.time()
    if time.time() - st.session_state.first_request_time > 86400:
        st.session_state.request_count = 0
        st.session_state.first_request_time = time.time()
    return st.session_state.request_count < MAX_REQUESTS_PER_SESSION


#most recent cv download

def _http_get(url, timeout, accept=None):
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _cv_cache_paths():
    return (
        os.path.join(CV_CACHE_DIR, "psi-ms.obo"),
        os.path.join(CV_CACHE_DIR, "cv_meta.json"),
    )


def _read_cv_disk_cache():
    obo_path, meta_path = _cv_cache_paths()
    if not (os.path.exists(obo_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(obo_path, "r", encoding="utf-8") as f:
            text = f.read()
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not text.strip():
            return None
        return text, meta
    except Exception as e:
        logger.warning("Failed to read CV disk cache: %s", e)
        return None


def _write_cv_disk_cache(text, meta):
    try:
        os.makedirs(CV_CACHE_DIR, exist_ok=True)
        obo_path, meta_path = _cv_cache_paths()
        with open(obo_path, "w", encoding="utf-8") as f:
            f.write(text)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.warning("Failed to write CV disk cache: %s", e)


def parse_obo_header(text):
    """pull format version, data version, date out of the OBO header"""
    header = text.split("[Term]", 1)[0]
    out = {}
    for key in ("format-version", "data-version", "date", "saved-by"):
        m = re.search(rf"^{re.escape(key)}:\s*(.+)$", header, re.M)
        if m:
            out[key] = m.group(1).strip()
    return out


def _resolve_cv_commit_sha():
    try:
        payload = json.loads(
            _http_get(
                OBO_COMMITS_API, CV_API_TIMEOUT,
                accept="application/vnd.github+json",
            )
        )
        if isinstance(payload, list) and payload:
            sha = payload[0].get("sha")
            if sha and re.fullmatch(r"[0-9a-f]{40}", sha):
                return sha
    except Exception as e:
        logger.warning("Could not resolve psi-ms.obo commit sha: %s", e)
    return None


@st.cache_data(ttl=CV_CACHE_TTL, show_spinner=False)
def fetch_cv_text(refresh_token=0):
    meta = {
        "origin": None,
        "retrieved_at": None,
        "source_url": OBO_RAW_URL,
        "blob_url": OBO_BLOB_URL,
        "commit_sha": None,
        "stable_uri": OBO_RAW_URL,
        "error": None,
    }
    text = None
    try:
        text = _http_get(OBO_RAW_URL, CV_DOWNLOAD_TIMEOUT)
        if "[Term]" not in text:
            raise ValueError("downloaded file does not look like an OBO file")
        meta["origin"] = "GitHub (master)"
        meta["retrieved_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="seconds")
        sha = _resolve_cv_commit_sha()
        if not sha:
            cached = _read_cv_disk_cache()
            if cached and cached[0] == text:
                sha = cached[1].get("commit_sha")
        if sha:
            meta["commit_sha"] = sha
            meta["stable_uri"] = OBO_PINNED_URL_TEMPLATE.format(sha=sha)
    except Exception as e:
        logger.warning("CV download failed: %s", e)
        meta["error"] = str(e)
        text = None

    if text is not None:
        meta.update(parse_obo_header(text))
        _write_cv_disk_cache(text, meta)
        return text, meta

    cached = _read_cv_disk_cache()
    if cached:
        text, cached_meta = cached
        cached_meta["origin"] = "local cache (download failed)"
        cached_meta["error"] = meta["error"]
        return text, cached_meta

    if os.path.exists(LOCAL_OBO_FALLBACK):
        with open(LOCAL_OBO_FALLBACK, "r", encoding="utf-8") as f:
            text = f.read()
        meta["origin"] = "psi-ms.obo next to the script (download failed)"
        meta.update(parse_obo_header(text))
        return text, meta

    return "", meta


#OBO parsing

def _new_term_record():
    return {
        "id": None, "name": None, "def": None, "def_xrefs": [],
        "comment": None, "synonyms": [], "xrefs": [], "is_a": [],
        "part_of": [],
        "is_obsolete": False, "replaced_by": None, "categories": [],
        "units": [], "xsd_value_type": None, "value_concepts": [],
        "columns": [], "optional_columns": [], "relations": [],
        "order": None, "domain": None,
    }


def parse_obo_text(text):
    """parse a full OBO document into a list of term dicts"""
    raw_blocks = re.split(r"\n(?=\[)", text.replace("\r\n", "\n"))
    terms = []
    for block in raw_blocks:
        stripped = block.strip()
        if not stripped.startswith("[Term]"):
            continue
        term = _new_term_record()
        for line in stripped.splitlines():
            line = line.strip()
            if not line or line.startswith("["):
                continue
            if line.startswith("id: "):
                term["id"] = line[4:].strip()
            elif line.startswith("name: "):
                term["name"] = line[6:].strip()
            elif line.startswith("def: "):
                m = re.match(r'def:\s*"(.*)"\s*\[([^\]]*)\]', line)
                if m:
                    term["def"] = m.group(1)
                    term["def_xrefs"] = [
                        r.strip() for r in m.group(2).split(",") if r.strip()
                    ]
                else:
                    term["def"] = line[5:].strip().strip('"')
            elif line.startswith("comment: "):
                term["comment"] = line[9:].strip()
            elif line.startswith("synonym: "):
                #keep scope plus xrefs so an exact synonym is not rewritten as related [] when the term is echoed back
                m = re.match(r'synonym:\s*"(.*?)"\s*(.*)$', line)
                if m:
                    term["synonyms"].append({
                        "name": m.group(1),
                        "rest": m.group(2).strip(),
                    })
            elif line.startswith("xref: "):
                term["xrefs"].append(line[6:].strip())
            elif line.startswith("is_a: "):
                term["is_a"].append(line[6:].strip())
            elif line.startswith("is_obsolete:"):
                term["is_obsolete"] = (
                    line.split(":", 1)[1].strip().lower() == "true"
                )
            elif line.startswith("replaced_by: "):
                term["replaced_by"] = line[13:].strip()
            elif line.startswith("relationship: "):
                _parse_relationship(term, line[14:].strip())
        if term["id"] and term["name"]:
            terms.append(term)
    return terms


def _parse_relationship(term, rel):
    """dispatch a single relationship line to correct term field"""
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
        #cv has trailing spaces on many of these lines strip them or "xsd:float " fails every downstream comparison
        term["xsd_value_type"] = rel[15:].strip()
    elif rel.startswith("has_order "):
        term["order"] = rel[10:].strip()
    elif rel.startswith("has_domain "):
        term["domain"] = rel[11:].strip()
    elif rel.startswith("part_of "):
        term["part_of"].append(rel[8:].strip())

def _numeric_ms_id(term_id):
    m = re.fullmatch(r"MS:(\d{7})", term_id or "")
    return int(m.group(1)) if m else None


def _is_in_qc_namespace(term_id):
    n = _numeric_ms_id(term_id)
    return n is not None and QC_NAMESPACE_MIN <= n <= QC_NAMESPACE_MAX


def _descendants_of(children_map, root_id, stop_at=()):
    """transitive children of root id"""
    stop = set(stop_at)
    seen, frontier = set(), [root_id]
    while frontier:
        cur = frontier.pop()
        if cur in stop and cur != root_id:
            continue
        for child in children_map.get(cur, ()):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return seen


def build_cv_index(raw_terms):
    """build every name->accession lookup the app needs straight from cv"""
    id_to_name = {t["id"]: t["name"] for t in raw_terms}
    id_to_term = {t["id"]: t for t in raw_terms}

    children = {}
    for t in raw_terms:
        for parent in t.get("is_a", []):
            pid = parent.split(" ! ")[0].strip()
            children.setdefault(pid, set()).add(t["id"])

    vtype_direct_children = set(children.get(QC_VALUE_TYPE_ROOT_ID, ()))
    vtype_name_to_id, vtype_id_to_name = {}, {}
    for vid in sorted(vtype_direct_children):
        vtype_name_to_id[id_to_name[vid]] = vid
        vtype_id_to_name[vid] = id_to_name[vid]
    if not vtype_name_to_id:
        vtype_name_to_id = dict(FALLBACK_VTYPE_NAME_TO_ID)
        vtype_id_to_name = {v: k for k, v in FALLBACK_VTYPE_NAME_TO_ID.items()}

    #anything under one of the four value types is a metric never a category
    metric_ids = set()
    for t in raw_terms:
        parents = {p.split(" ! ")[0].strip() for p in t.get("is_a", [])}
        if parents & vtype_direct_children:
            metric_ids.add(t["id"])

    category_ids = set(_descendants_of(children, QC_CATEGORY_ROOT_ID))
    category_ids |= _descendants_of(
        children, QC_METRIC_ID, stop_at={QC_VALUE_TYPE_ROOT_ID}
    )
    for t in raw_terms:
        for cat in t.get("categories", []):
            cid = cat.split(" ! ")[0].strip()
            if cid in id_to_name:
                category_ids.add(cid)
    category_ids -= vtype_direct_children
    category_ids -= metric_ids
    category_ids -= {QC_ROOT_ID, QC_METRIC_ID, QC_VALUE_TYPE_ROOT_ID,
                     QC_CATEGORY_ROOT_ID}
    category_name_to_id = {
        id_to_name[c]: c for c in sorted(category_ids) if c in id_to_name
    }
    if not category_name_to_id:
        category_name_to_id = dict(FALLBACK_CATEGORY_NAME_TO_ID)
    if len(category_name_to_id) > 80:
        #tripwire: if this fires the category anchors in the CV have moved
        logger.warning(
            "Category allow list is unusually large (%d entries), the CV "
            "hierarchy may have changed", len(category_name_to_id)
        )

    unit_id_to_name = {}
    for t in raw_terms:
        for u in t.get("units", []):
            parts = u.split(" ! ", 1)
            uid = parts[0].strip()
            uname = (
                parts[1].strip() if len(parts) == 2
                else id_to_name.get(uid, uid)
            )
            unit_id_to_name.setdefault(uid, uname)
    for uid, uname in EXTRA_UNIT_NAMES.items():

        unit_id_to_name.setdefault(uid, id_to_name.get(uid, uname))

    unit_name_to_id = {}
    for uid, uname in unit_id_to_name.items():
        unit_name_to_id[uname.lower()] = uid
    for alias, uid in UNIT_ALIAS_TO_ID.items():
        if uid not in unit_id_to_name:
            canonical = id_to_name.get(uid) or EXTRA_UNIT_NAMES.get(uid)
            if not canonical:
                logger.warning(
                    "Unit alias '%s' resolves to %s which has no canonical "
                    "name, dropping it from the allow list", alias, uid
                )
                continue
            unit_id_to_name[uid] = canonical
            unit_name_to_id.setdefault(canonical.lower(), uid)
        unit_name_to_id.setdefault(alias.lower(), uid)

    concept_id_to_name = {}
    for t in raw_terms:
        for vc in t.get("value_concepts", []):
            parts = vc.split(" ! ", 1)
            cid = parts[0].strip()
            cname = parts[1].strip() if len(parts) == 2 else cid
            concept_id_to_name.setdefault(cid, cname)

    #candidate columns: anything already used as one, plus any term carrying a
    #unit or typed as a single value metric (spec 4.1 allows both)
    column_id_to_name = {}
    used_column_id_to_name = {}
    single_value_id = vtype_name_to_id.get("single value")
    for t in raw_terms:
        for col in t.get("columns", []) + t.get("optional_columns", []):
            parts = col.split(" ! ", 1)
            cid = parts[0].strip()
            cname = (
                parts[1].strip() if len(parts) == 2
                else id_to_name.get(cid, cid)
            )
            #the CV uses NCIT plus OBI accessions as columns which psi-ms.obo
            #never defines, so resolve_column looks here first or a legal
            #column is dropped as unverifiable
            column_id_to_name.setdefault(cid, cname)
            used_column_id_to_name.setdefault(cid, cname)
    for t in raw_terms:
        if t.get("is_obsolete"):
            continue
        parents = [p.split(" ! ")[0].strip() for p in t.get("is_a", [])]
        if t.get("units") or (single_value_id and single_value_id in parents):
            column_id_to_name.setdefault(t["id"], t["name"])
    #terms under 'QC non-metric term' exist to be used as columns so they are
    #always offered, even when nothing references them yet
    for cid in _descendants_of(children, QC_NON_METRIC_ID):
        if cid in id_to_name:
            column_id_to_name.setdefault(cid, id_to_name[cid])
            used_column_id_to_name.setdefault(cid, id_to_name[cid])

    name_to_id_lower = {}
    for tid, tname in id_to_name.items():
        name_to_id_lower.setdefault(tname.lower(), tid)

    non_metric_parent_id = name_to_id_lower.get(
        QC_NON_METRIC_NAME.lower(), QC_NON_METRIC_ID
    )

    return {
        "id_to_name": id_to_name,
        "id_to_term": id_to_term,
        "name_to_id_lower": name_to_id_lower,
        "vtype_name_to_id": vtype_name_to_id,
        "vtype_id_to_name": vtype_id_to_name,
        "category_name_to_id": category_name_to_id,
        "unit_name_to_id": unit_name_to_id,
        "unit_id_to_name": unit_id_to_name,
        "concept_id_to_name": concept_id_to_name,
        "column_id_to_name": column_id_to_name,
        "used_column_id_to_name": used_column_id_to_name,
        "metric_ids": metric_ids,
        "non_metric_parent_id": non_metric_parent_id,
        "non_metric_parent_name": id_to_name.get(
            non_metric_parent_id, QC_NON_METRIC_NAME
        ),
        "existing_names_lower": {
            t["name"].lower() for t in raw_terms if not t.get("is_obsolete")
        },
    }


def resolve_unit(cv_index, unit_name):
    """resolve a unit name or accession to (id, canonical CV name)"""
    if not unit_name:
        return None, None
    raw = str(unit_name).strip()
    if raw in cv_index["unit_id_to_name"]:
        return raw, cv_index["unit_id_to_name"][raw]
    lower = raw.lower()
    uid = cv_index["unit_name_to_id"].get(lower)
    if not uid and lower.endswith("s") and len(lower) > 2:
        uid = cv_index["unit_name_to_id"].get(lower[:-1])
    if not uid:
        uid = cv_index["name_to_id_lower"].get(lower)
    if not uid:
        return None, None
    canonical = (
        cv_index["unit_id_to_name"].get(uid)
        or cv_index["id_to_name"].get(uid)
    )
    if not canonical:
        return None, None
    return uid, canonical


def resolve_category(cv_index, cat_name):
    if not cat_name:
        return None, None
    raw = str(cat_name).strip()
    by_id = {v: k for k, v in cv_index["category_name_to_id"].items()}
    if raw in by_id:
        return raw, by_id[raw]
    for name, cid in cv_index["category_name_to_id"].items():
        if name.lower() == raw.lower():
            return cid, name
    return None, None


def resolve_value_concept(cv_index, ref):
    if isinstance(ref, dict):
        candidate_id = (ref.get("id") or "").strip()
        candidate_name = (ref.get("name") or "").strip()
    else:
        candidate_id, candidate_name = "", str(ref or "").strip()
    concepts = cv_index["concept_id_to_name"]
    if candidate_id and candidate_id in concepts:
        return candidate_id, concepts[candidate_id]
    if candidate_name:
        for cid, cname in concepts.items():
            if cname.lower() == candidate_name.lower():
                return cid, cname
    #a concept that is a genuine psi-ms term (MS:1000086 FWHM) still resolves
    return resolve_cv_term(cv_index, ref)


def column_units(cv_index, column_id):
    if column_id in cv_index["unit_id_to_name"]:
        return [(column_id, cv_index["unit_id_to_name"][column_id])]
    term = cv_index["id_to_term"].get(column_id)
    if not term:
        return []
    out = []
    for u in term.get("units", []):
        parts = u.split(" ! ", 1)
        uid = parts[0].strip()
        uname = (
            parts[1].strip() if len(parts) == 2
            else cv_index["unit_id_to_name"].get(uid, uid)
        )
        out.append((uid, uname))
    return out


def resolve_column(cv_index, ref):
    """resolve a table column term. the harvested column set is checked 1st bc the CV uses accessions from other ontologies as columns"""
    if isinstance(ref, dict):
        candidate_id = (ref.get("id") or "").strip()
        candidate_name = (ref.get("name") or "").strip()
    else:
        candidate_id, candidate_name = "", str(ref or "").strip()
    cols = cv_index["column_id_to_name"]
    if candidate_id and candidate_id in cols:
        return candidate_id, cols[candidate_id]
    if candidate_name:
        for cid, cname in cols.items():
            if cname.lower() == candidate_name.lower():
                return cid, cname
    return resolve_cv_term(cv_index, ref)


def resolve_cv_term(cv_index, ref):
    if isinstance(ref, dict):
        candidate_id = (ref.get("id") or "").strip()
        candidate_name = (ref.get("name") or "").strip()
    else:
        candidate_id, candidate_name = "", str(ref or "").strip()
    if candidate_id and candidate_id in cv_index["id_to_name"]:
        return candidate_id, cv_index["id_to_name"][candidate_id]
    if candidate_name:
        tid = cv_index["name_to_id_lower"].get(candidate_name.lower())
        if tid:
            return tid, cv_index["id_to_name"][tid]
    return None, None


#knowledge graph

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
        for part_ref in t.get("part_of", []):
            part_id = part_ref.split(" ! ")[0].strip()
            if part_id in G:
                G.add_edge(tid, part_id, rel="part_of")
        for rel_ref in t.get("relations", []):
            rel_id = rel_ref.split(" ! ", 1)[0].strip()
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
        for col_ref in t.get("columns", []) + t.get("optional_columns", []):
            col_id = col_ref.split(" ! ")[0].strip()
            if col_id in G:
                G.add_edge(tid, col_id, rel="has_column")
    return G


def get_graph_neighbors(G, node_ids, hops=1, hub_degree_limit=HUB_DEGREE_LIMIT):
    """breadth first expansion returning {node_id: hop_distance} for the neighbours only. expansion never continues through a hub node, one hop from MS:4000003 would otherwise return the whole namespace"""
    dist = {n: 0 for n in node_ids if n in G}
    frontier = set(dist)
    for hop in range(1, hops + 1):
        next_frontier = set()
        for n in frontier:
            if n not in G:
                continue
            if hop > 1 and (G.in_degree(n) + G.out_degree(n)) > hub_degree_limit:
                continue
            next_frontier |= set(G.successors(n))
            next_frontier |= set(G.predecessors(n))
        next_frontier -= set(dist)
        for n in next_frontier:
            dist[n] = hop
        frontier = next_frontier
    return {n: d for n, d in dist.items() if d > 0}


#embeddings

def _term_embedding_text(term):
    parts = [term.get("name", "")]
    if term.get("def"):
        parts.append(term["def"])
    if term.get("comment"):
        parts.append(term["comment"])
    for syn in term.get("synonyms", []):
        parts.append(syn["name"] if isinstance(syn, dict) else str(syn))
    cats = [c.split(" ! ")[-1] for c in term.get("categories", [])]
    if cats:
        parts.append("Categories: " + ", ".join(cats))
    parents = [p.split(" ! ")[-1] for p in term.get("is_a", [])]
    if parents:
        parts.append("Is-a: " + ", ".join(parents))
    part_ofs = [p.split(" ! ")[-1] for p in term.get("part_of", [])]
    if part_ofs:
        parts.append("Part-of: " + ", ".join(part_ofs))
    return ". ".join(parts)


def _obo_content_hash(raw_terms):
    content = json.dumps(
        [(t["id"], t["name"], t.get("def", "")) for t in raw_terms],
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_embedding_cache(obo_hash):
    meta_path = os.path.join(EMBEDDING_CACHE_DIR, "meta.json")
    matrix_path = os.path.join(EMBEDDING_CACHE_DIR, "embeddings.npz")
    if not (os.path.exists(meta_path) and os.path.exists(matrix_path)):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("cache_version") != EMBEDDING_CACHE_VERSION:
            return None
        if meta.get("obo_hash") != obo_hash:
            return None
        data = np.load(matrix_path)
        matrix = data["embeddings"]
        if not np.all(np.isfinite(matrix)):
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
    embedding_matrix, api_key, embed_quota=EMBED_QUOTA,
    graph_quota=GRAPH_QUOTA, hops=GRAPH_HOP_LIMIT,
):
    query_text = f"{query_name}. {query_desc}"
    q_emb = embed_query(query_text, api_key)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = (
            q_emb.astype(np.float64) @ embedding_matrix.astype(np.float64).T
        ).flatten()
    sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
    id_to_sim = {term_ids[i]: float(sims[i]) for i in range(len(term_ids))}

    top_indices = np.argsort(sims)[::-1][:embed_quota]
    seed_ids = [term_ids[i] for i in top_indices]

    neighbor_dist = get_graph_neighbors(G, set(seed_ids), hops=hops) if G else {}
    neighbors = sorted(
        (n for n in neighbor_dist if n not in set(seed_ids)),
        key=lambda n: (neighbor_dist[n], -id_to_sim.get(n, -1.0)),
    )[:graph_quota]

    id_to_term = {t["id"]: t for t in raw_terms}
    ordered = seed_ids + neighbors
    return [id_to_term[tid] for tid in ordered if tid in id_to_term]


#term helpers

def derive_value_type_from_is_a(term, cv_index):
    vtype_id_to_name = cv_index["vtype_id_to_name"]
    for parent in term.get("is_a", []):
        pid = parent.split(" ! ")[0].strip()
        if pid in vtype_id_to_name:
            return vtype_id_to_name[pid]
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


def _is_qc_metric_term(term_display):
    """determine whether a display term is a QC metric vs base MS term"""
    if term_display.get("value_type"):
        return True
    if _is_in_qc_namespace(term_display.get("id", "")):
        #QC non metric terms live in the QC namespace but are columns, never duplicate candidates
        parents = term_display.get("is_a", [])
        parent_ids = {p["id"] for p in parents if isinstance(p, dict)}
        if QC_NON_METRIC_ID in parent_ids:
            return False
        return True
    if term_display.get("categories"):
        return True
    return False


def term_to_display_dict(term, cv_index):
    vtype = derive_value_type_from_is_a(term, cv_index)
    cats = [extract_name(c) for c in term.get("categories", [])]
    units = [extract_name(u) for u in term.get("units", [])]
    concepts = [extract_name(v) for v in term.get("value_concepts", [])]
    relations = [extract_id_name(r) for r in term.get("relations", [])]
    columns = [extract_name(c) for c in term.get("columns", [])]
    opt_columns = [extract_name(c) for c in term.get("optional_columns", [])]
    parents = [extract_id_name(p) for p in term.get("is_a", [])]
    part_ofs = [extract_id_name(p) for p in term.get("part_of", [])]
    d = {"id": term["id"], "name": term["name"], "def": term.get("def")}
    if term.get("comment"):
        d["comment"] = term["comment"]
    if term.get("synonyms"):
        d["synonyms"] = [
            s["name"] if isinstance(s, dict) else str(s)
            for s in term["synonyms"]
        ]
    if term.get("xrefs"):
        d["xrefs"] = term["xrefs"]
    if parents:
        d["is_a"] = parents
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
    if part_ofs:
        d["part_of"] = part_ofs
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


def compute_verdict_flags(result):
    """single source of truth for the verdict booleans, used by the render layer as well as the branch that persists accepted metrics"""
    top_level = max_overlap_level(result.get("overlap_results", []))
    is_duplicate = top_level == "duplicate"
    is_high = top_level == "high"
    is_new_metric = (
        not result.get("needs_more_detail")
        and bool(result.get("is_new"))
        and top_level not in ("duplicate", "high")
    )
    return top_level, is_duplicate, is_high, is_new_metric


def find_raw_term_by_id(raw_terms, term_id):
    for t in raw_terms:
        if t["id"] == term_id:
            return t
    return None


#text sanitation, spec 4.1

def _sanitize_cv_text(text):
    """spec 4.1: no escaped characters or backticks in a name, definition or comment, single quotes quote words. newlines are flattened because a def must live on ONE obo line or the pasted block corrupts psi-ms.obo"""
    if not text:
        return ""
    t = str(text)
    t = t.replace('\\"', "'").replace('"', "'").replace("`", "'")
    t = t.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _strip_def_boilerplate(text):
    stripped = DEF_BOILERPLATE_RE.sub("", text or "", count=1)
    if stripped != text and stripped:
        stripped = stripped[0].upper() + stripped[1:]
    return stripped


def _clean_def_text(text):
    text = (text or "").strip()
    while len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    text = _sanitize_cv_text(text)
    text = _strip_def_boilerplate(text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _fix_name_casing(name):
    if not name:
        return name
    words = name.split()
    fixed = []
    for w in words:
        key = w.lower()
        if key in MS_ABBREVIATIONS:
            fixed.append(MS_ABBREVIATIONS[key])
        elif "-" in w:
            parts = w.split("-")
            parts_fixed = [
                MS_ABBREVIATIONS.get(p.lower(), p) for p in parts
            ]
            fixed.append("-".join(parts_fixed))
        else:
            fixed.append(w)
    return " ".join(fixed)


NAME_STOPWORDS = {
    "a", "an", "the", "of", "for", "per", "in", "on", "and", "or", "to",
    "by", "from", "with", "based", "metric", "value", "values",
}


def _name_tokens(name):
    return {
        w for w in re.split(r"[^a-z0-9/]+", (name or "").lower())
        if w and w not in NAME_STOPWORDS
    }


def name_drift_ratio(proposed_name, drafted_name):
    """jaccard overlap of the content words of two names. a low value means the drafted term is not the metric that was proposed. returns 1.0 when there is nothing to compare"""
    a, b = _name_tokens(proposed_name), _name_tokens(drafted_name)
    if not a or not b:
        return 1.0
    return len(a & b) / len(a | b)


def normalize_value_type(vtype):
    """map whatever the model wrote onto a CV value type name"""
    v = (vtype or "single value").strip().lower()
    v = VTYPE_ALIASES.get(v, v)
    for canonical in ("single value", "n-tuple", "table", "matrix"):
        if v == canonical:
            return canonical
    return "single value"


def _clean_xref(value):
    """an OBO xref is one token: PMID:123, DOI:10.x/y or a URL. anything with a space or a bracket would break the stanza"""
    x = _sanitize_cv_text(value).strip()
    if not x or " " in x or "[" in x or "]" in x:
        return ""
    return x


#OBO generation and spec validation

def resolve_term_parts(result, cv_index):
    """resolve every accession the model proposed against the live cv exactly once, so the OBO block, the compliance report, the mzQC snippet all work from the same values"""
    vtype = normalize_value_type(result.get("metric_value_type"))
    vtype_id = cv_index["vtype_name_to_id"].get(vtype)

    units, unknown_units = [], []
    for u in result.get("suggested_units") or []:
        uid, uname = resolve_unit(cv_index, u)
        if uid:
            if uid not in [x[0] for x in units]:
                units.append((uid, uname))
        else:
            unknown_units.append(str(u))

    categories, unknown_categories = [], []
    for c in result.get("suggested_categories") or []:
        cid, cname = resolve_category(cv_index, c)
        if cid:
            if cid not in [x[0] for x in categories]:
                categories.append((cid, cname))
        else:
            unknown_categories.append(str(c))

    concept = (None, None)
    raw_concept = result.get("suggested_value_concept")
    if raw_concept:
        cid, cname = resolve_value_concept(cv_index, raw_concept)
        if cid:
            concept = (cid, cname)

    columns, optional_columns, unknown_columns = [], [], []
    for col in result.get("suggested_columns") or []:
        cid, cname = resolve_column(cv_index, col)
        if not cid:
            unknown_columns.append(
                col.get("name") if isinstance(col, dict) else str(col)
            )
            continue
        is_optional = (
            bool(col.get("optional")) if isinstance(col, dict) else False
        )
        (optional_columns if is_optional else columns).append((cid, cname))

    #columns that do not exist yet are NOT dropped, each gets a draft [Term] block under QC non-metric term with a numbered placeholder accession
    new_columns = []
    for i, nc in enumerate(result.get("suggested_new_column_terms") or [], 1):
        if not isinstance(nc, dict):
            nc = {"name": str(nc)}
        cname = _fix_name_casing(_sanitize_cv_text(nc.get("name")))
        if not cname:
            continue
        uid, uname = resolve_unit(cv_index, nc.get("unit"))
        xsd = (nc.get("xsd_type") or "").strip()
        if xsd not in ALLOWED_XSD_TYPES:
            xsd = "xsd:float"
        #a statistical column (R2, slope, intercept, CV, mean) carries the STATO/NCIT concept it represents, like the metric itself
        ccid, ccname = (None, None)
        if nc.get("value_concept"):
            ccid, ccname = resolve_value_concept(
                cv_index, nc.get("value_concept")
            )
        new_columns.append({
            "concept_id": ccid,
            "concept_name": ccname,
            "placeholder_id": NEW_COLUMN_PLACEHOLDER_TEMPLATE.format(n=i),
            "name": cname,
            "def": _clean_def_text(nc.get("def")),
            "xsd_type": xsd,
            "unit_id": uid,
            "unit_name": uname,
            "optional": bool(nc.get("optional")),
            "xrefs": [
                x for x in (_clean_xref(v) for v in (nc.get("xrefs") or []))
                if x
            ],
        })

    relations = []
    for rel in result.get("suggested_relations") or []:
        rid, rname = resolve_cv_term(cv_index, rel)
        if rid and rid not in [x[0] for x in relations]:
            relations.append((rid, rname))

    xsd_type = (result.get("suggested_xsd_type") or "").strip()
    if xsd_type and xsd_type not in ALLOWED_XSD_TYPES:
        xsd_type = "xsd:float" if xsd_type.startswith("xsd:") else ""

    synonyms = []
    for syn in result.get("suggested_synonyms") or []:
        if isinstance(syn, dict):
            sname = _sanitize_cv_text(syn.get("name"))
            scope = (syn.get("scope") or "RELATED").upper()
            xref = _clean_xref(syn.get("xref"))
        else:
            sname, scope, xref = _sanitize_cv_text(syn), "RELATED", ""
        if scope not in ("EXACT", "NARROW", "BROAD", "RELATED"):
            scope = "RELATED"
        if sname:
            synonyms.append((sname, scope, xref))

    xrefs = []
    for x in result.get("suggested_xrefs") or []:
        cleaned = _clean_xref(x)
        if cleaned and cleaned not in xrefs:
            xrefs.append(cleaned)

    return {
        "name": _fix_name_casing(
            _sanitize_cv_text(result.get("suggested_name"))
        ),
        "def": _clean_def_text(result.get("suggested_def")),
        "comment": _sanitize_cv_text(result.get("suggested_comment")),
        "value_type": vtype,
        "value_type_id": vtype_id,
        "xsd_type": xsd_type,
        "units": units,
        "unknown_units": unknown_units,
        "categories": categories,
        "unknown_categories": unknown_categories,
        "value_concept": concept,
        "columns": columns,
        "optional_columns": optional_columns,
        "unknown_columns": unknown_columns,
        "new_columns": new_columns,
        "relations": relations,
        "synonyms": synonyms,
        "xrefs": xrefs,
    }


def generate_obo_block(result, cv_index):
    """generate a new OBO [Term] block. every accession in it is resolved against the downloaded CV so the block can be pasted into a term request without hand checking the numbers"""
    p = resolve_term_parts(result, cv_index)
    lines = [
        "[Term]",
        f"id: {NEW_TERM_PLACEHOLDER}",
        f"name: {p['name']}",
        f'def: "{p["def"]}" [PSI:MS]',
    ]
    if p["comment"]:
        lines.append(f"comment: {p['comment']}")
    for sname, scope, xref in p["synonyms"]:
        lines.append(f'synonym: "{sname}" {scope} [{xref}]')
    for x in p["xrefs"]:
        lines.append(f"xref: {x}")
    if p["value_type_id"]:
        lines.append(f"is_a: {p['value_type_id']} ! {p['value_type']}")
    if p["xsd_type"]:
        lines.append(
            f"relationship: has_value_type {p['xsd_type']}"
            " ! The allowed value-type for this CV term"
        )
    #spec 4.1: a table carries no unit of its own, its units come from the has_units of each column term
    if p["value_type"] != "table":
        for uid, uname in p["units"]:
            lines.append(f"relationship: has_units {uid} ! {uname}")
    if p["value_concept"][0]:
        lines.append(
            f"relationship: has_value_concept {p['value_concept'][0]}"
            f" ! {p['value_concept'][1]}"
        )
    for cid, cname in p["categories"]:
        lines.append(f"relationship: has_metric_category {cid} ! {cname}")
    for cid, cname in p["columns"]:
        lines.append(f"relationship: has_column {cid} ! {cname}")
    for nc in p["new_columns"]:
        if not nc["optional"]:
            lines.append(
                f"relationship: has_column {nc['placeholder_id']}"
                f" ! {nc['name']}"
            )
    for cid, cname in p["optional_columns"]:
        lines.append(f"relationship: has_optional_column {cid} ! {cname}")
    for nc in p["new_columns"]:
        if nc["optional"]:
            lines.append(
                f"relationship: has_optional_column {nc['placeholder_id']}"
                f" ! {nc['name']}"
            )
    for rid, rname in p["relations"]:
        lines.append(f"relationship: has_relation {rid} ! {rname}")
    return "\n".join(lines), p


def generate_new_column_blocks(parts, cv_index):
    """draft [Term] blocks for the table columns that do not exist yet. each is is_a QC non-metric term: a column is not a metric, a term with no is_a floats at the top level of the CV"""
    parent_id = cv_index["non_metric_parent_id"]
    parent_name = cv_index["non_metric_parent_name"]
    blocks = []
    for nc in parts["new_columns"]:
        lines = [
            "[Term]",
            f"id: {nc['placeholder_id']}",
            f"name: {nc['name']}",
            f'def: "{nc["def"] or "TODO: define this column term."}" [PSI:MS]',
        ]
        for x in nc["xrefs"]:
            lines.append(f"xref: {x}")
        lines.append(f"is_a: {parent_id} ! {parent_name}")
        lines.append(
            f"relationship: has_value_type {nc['xsd_type']}"
            " ! The allowed value-type for this CV term"
        )
        if nc.get("concept_id"):
            lines.append(
                f"relationship: has_value_concept {nc['concept_id']}"
                f" ! {nc['concept_name']}"
            )
        if nc["unit_id"]:
            lines.append(
                f"relationship: has_units {nc['unit_id']} ! {nc['unit_name']}"
            )
        blocks.append((nc["placeholder_id"], nc["name"], "\n".join(lines)))
    return blocks


IDENTIFIER_NAME_RE = re.compile(
    r"\b(identifier|id|accession|name|key|label)\b", re.IGNORECASE
)

#words that show the definition says what the identifier actually is
FORMAT_HINT_RE = re.compile(
    r"(accurate mass|retention time|\bm/z\b|transition|precursor|"
    r"accession|USI|spectral librar|database|InChI|SMILES|formula|CAS|"
    r"LIPID MAPS|HMDB|ChEBI|UniProt|proforma|scan number|index|"
    r"native spectrum identifier)", re.IGNORECASE
)


def _is_identifier_column(new_column):
    """a column that names the row rather than measuring it"""
    if new_column.get("xsd_type") != "xsd:string":
        return False
    return bool(IDENTIFIER_NAME_RE.search(new_column.get("name", "")))


def _states_a_format(definition):
    return bool(FORMAT_HINT_RE.search(definition or ""))


def validate_generated_term(parts, cv_index, proposed_name=None):
    """check the drafted term against the MUST/SHOULD rules of mzQC 1.0.0 section 4.1 plus section 7. returns the list the compliance panel renders"""
    checks = []

    def add(level, rule, message):
        checks.append({"level": level, "rule": rule, "message": message})

    name = parts["name"]
    if not name:
        add("error", "4.1 Name", "No name was generated.")
    else:
        if len(name) > MAX_NAME_LENGTH:
            add("warning", "4.1 Name",
                f"Name is {len(name)} characters, the specification says it "
                f"SHOULD stay under {MAX_NAME_LENGTH}.")
        if not name.isascii():
            add("error", "4.1 Name",
                "Name contains non 7-bit-ASCII characters.")
        elif not NAME_ALLOWED_RE.match(name):
            bad = sorted({ch for ch in name if not NAME_ALLOWED_RE.match(ch)})
            add("warning", "4.1 Name",
                "Name uses characters outside the recommended set "
                "[alphanumeric, space, - _ , .]: " + " ".join(bad) +
                ". Several existing CV terms do the same (m/z), so this is "
                "acceptable if intentional.")
        if name.lower() in cv_index["existing_names_lower"]:
            add("error", "4.1 Name",
                f"An active CV term is already called '{name}'. Names must be "
                "unambiguous, pick a different one or reuse the existing term.")
        elif len(name) <= MAX_NAME_LENGTH and name.isascii():
            add("ok", "4.1 Name",
                f"'{name}' is unique in this CV version and within the length "
                "limit.")
        #drafted term must be the metric that was proposed a renamed one usually means the model answered a different question
        if proposed_name and name_drift_ratio(proposed_name, name) < 0.6:
            add("warning", "Proposal fidelity",
                f"The drafted term is called '{name}' but you proposed "
                f"'{proposed_name}'. Check that this is still the metric you "
                "asked about: the OBO block, value type and scope below "
                "describe the drafted name, not necessarily your proposal. "
                "Redesign advice belongs in the consolidation suggestion.")

    if not parts["def"]:
        add("error", "4.1 Definition", "The definition is empty.")
    else:
        if '"' in parts["def"] or "`" in parts["def"]:
            add("error", "4.1 Definition",
                "Definition still contains double quotes or backticks.")
        elif DEF_BOILERPLATE_RE.match(parts["def"]):
            add("warning", "4.1 Definition",
                "Definition still opens with a 'quality control metric "
                "reporting' style filler clause, state directly what the "
                "value is.")
        else:
            add("ok", "4.1 Definition",
                "Definition is a single quoted line with a PSI:MS xref and no "
                "filler opener.")

    if parts["comment"]:
        if parts["comment"].lower().startswith(parts["def"].lower()[:40]):
            add("warning", "4.1 Comment",
                "Comment appears to repeat the definition, the specification "
                "says a comment should add non-trivial information instead.")
        else:
            add("ok", "4.1 Comment", "Comment adds interpretation guidance.")

    if not parts["value_type_id"]:
        add("error", "4.1 Value type",
            "The value type could not be mapped to a CV term "
            "(single value, n-tuple, table, matrix).")
    else:
        add("ok", "4.1 Value type",
            f"is_a {parts['value_type_id']} ! {parts['value_type']}")

    if parts["value_type"] == "table":
        n_existing = len(parts["columns"]) + len(parts["optional_columns"])
        n_new = len(parts["new_columns"])
        if n_existing == 0 and n_new == 0:
            add("error", "4.1 Table columns",
                "A table MUST define at least one has_column term, and none "
                "were resolved or proposed.")
        elif n_existing:
            add("ok", "4.1 Table columns",
                f"{len(parts['columns'])} required column(s) resolved"
                + (f", {len(parts['optional_columns'])} optional"
                   if parts["optional_columns"] else "")
                + (f", plus {n_new} drafted below" if n_new else ""))
        else:
            add("ok", "4.1 Table columns",
                f"No existing CV term fits any column, all {n_new} are "
                "drafted below as new QC non-metric terms.")
        if n_new:
            add("warning", "4.1 Table columns",
                f"{n_new} column term(s) do not exist in the CV yet and are "
                "drafted below as separate QC non-metric terms: "
                + ", ".join(nc["name"] for nc in parts["new_columns"])
                + ". They must be accepted and assigned real accessions "
                "before the placeholder ids in the has_column lines can be "
                "replaced.")
        if parts["units"]:
            add("warning", "4.1 Units",
                "Units were suggested for a table metric, they were dropped: "
                "table units come from the column terms.")
        for cid, cname in parts["columns"] + parts["optional_columns"]:
            cu = column_units(cv_index, cid)
            if len(cu) > 1:
                add("warning", "4.1 Table columns",
                    f"Column {cid} ({cname}) declares more than one unit "
                    + ", ".join(f"{u[0]} {u[1]}" for u in cu)
                    + ". The specification asks you to state in the table "
                    "definition which one this metric expects.")
            elif not cu:
                add("warning", "4.1 Table columns",
                    f"Column {cid} ({cname}) declares no has_units, so its "
                    "unit is not machine readable. That is fine for "
                    "identifier and label columns, otherwise pick a column "
                    "term that carries a unit.")
        for nc in parts["new_columns"]:
            if not nc["def"]:
                add("warning", "4.1 Table columns",
                    f"Proposed column term '{nc['name']}' has no definition.")
            elif _is_identifier_column(nc) and not _states_a_format(nc["def"]):
                add("warning", "4.1 Table columns",
                    f"Proposed identifier column '{nc['name']}' does not "
                    "say what identifies the row. State the format "
                    "(accurate mass and retention time, a transition m/z "
                    "pair, a database accession, a specified name): there "
                    "is no feature-level USI yet, so an unspecified "
                    "identifier is not reproducible across tools.")
    else:
        if parts["new_columns"]:
            add("warning", "4.1 Table columns",
                "New column terms were proposed for a non-table metric and "
                "were ignored.")
        if not parts["units"]:
            add("error", "4.1 Units",
                "has_units is REQUIRED for single value, n-tuple and matrix "
                "metrics. Nothing resolvable was supplied"
                + (f" (unrecognised: {', '.join(parts['unknown_units'])})"
                   if parts["unknown_units"] else "")
                + ". If the quantity is dimensionless use "
                "UO:0000186 ! dimensionless unit.")
        elif len(parts["units"]) > 1:
            add("error", "4.1 Units",
                "Only one uniform unit is allowed for this value type, "
                f"{len(parts['units'])} were suggested.")
        else:
            add("ok", "4.1 Units",
                f"has_units {parts['units'][0][0]} ! {parts['units'][0][1]}")

    if parts["xsd_type"]:
        if parts["xsd_type"] in QC_PREFERRED_XSD_TYPES:
            add("ok", "4.1 Value type detail",
                f"has_value_type {parts['xsd_type']}")
        else:
            add("warning", "4.1 Value type detail",
                f"has_value_type {parts['xsd_type']} is valid but the QC "
                "namespace only uses "
                + ", ".join(QC_PREFERRED_XSD_TYPES)
                + ". Use xsd:int for counts and xsd:float for continuous "
                "quantities to stay consistent with the neighbouring terms.")
    else:
        add("warning", "4.1 Value type detail",
            "has_value_type is RECOMMENDED and is missing or was not "
            "recognised.")

    cat_names = [c[1] for c in parts["categories"]]
    if not any(c in ("ID based metric", "ID free metric") for c in cat_names):
        add("warning", "4.1 Categorization",
            "It is RECOMMENDED to state whether the metric is an "
            "'ID based metric' or an 'ID free metric'.")
    else:
        add("ok", "4.1 Categorization", ", ".join(cat_names))

    #a table with one row per feature summarising several runs is a multi run metric
    if parts["value_type"] == "table" and not any(
        c in ("multiple runs based metric", "multiple spectra based metric")
        for c in cat_names
    ):
        add("warning", "4.1 Categorization",
            "A table reports many rows. If the rows come from several runs add "
            "'multiple runs based metric', if they come from many spectra of "
            "one run add 'single run based metric' and 'multiple spectra "
            "based metric'.")

    if parts["unknown_categories"]:
        add("warning", "4.1 Categorization",
            "Dropped categories that are not in the CV: "
            + ", ".join(parts["unknown_categories"]))
    if parts["unknown_columns"]:
        add("warning", "4.1 Table columns",
            "Dropped columns that are neither in the CV nor drafted as new "
            "terms: " + ", ".join(str(c) for c in parts["unknown_columns"]))

    if parts["value_concept"][0]:
        add("ok", "4.1 Value concept",
            f"has_value_concept {parts['value_concept'][0]} "
            f"! {parts['value_concept'][1]}")
    else:
        add("warning", "4.1 Value concept",
            "has_value_concept is RECOMMENDED for full semantic integration "
            "(for example STATO:0000401 sample mean).")

    return checks


def generate_mzqc_snippet(parts, cv_index):
    snippet = {
        "accession": NEW_TERM_PLACEHOLDER,
        "name": parts["name"] or "new metric",
        "description": parts["def"],
    }
    if parts["value_type"] == "table":
        cols = [
            (cid, cname, column_units(cv_index, cid))
            for cid, cname in parts["columns"] + parts["optional_columns"]
        ]
        for nc in parts["new_columns"]:
            cu = (
                [(nc["unit_id"], nc["unit_name"])] if nc["unit_id"] else []
            )
            cols.append((nc["placeholder_id"], nc["name"], cu))
        snippet["value"] = {name: [] for _, name, _ in cols} or {"column": []}
        units = []
        for _, _, cu in cols:
            if len(cu) == 1:
                units.append({"accession": cu[0][0], "name": cu[0][1]})
            elif len(cu) > 1:
                units.append({"accession": cu[0][0], "name": cu[0][1],
                              "_ambiguous": [u[0] for u in cu]})
            else:
                units.append(None)
        snippet["unit"] = units
    else:
        if parts["xsd_type"] in ("xsd:int", "xsd:integer",
                                 "xsd:nonNegativeInteger",
                                 "xsd:positiveInteger"):
            base = 0
        elif parts["xsd_type"] == "xsd:string":
            base = "value"
        elif parts["xsd_type"] == "xsd:boolean":
            base = True
        else:
            #base is always a SCALAR, the ntuple or matrix shapes are built from it below
            base = 0.0
        if parts["value_type"] == "single value":
            snippet["value"] = base
        elif parts["value_type"] == "n-tuple":
            snippet["value"] = [base, base, base]
        else:
            snippet["value"] = [[base, base], [base, base]]
        if parts["units"]:
            uid, uname = parts["units"][0]
            snippet["unit"] = {"accession": uid, "name": uname}
    return json.dumps(snippet, indent=2)


def generate_cv_reference_snippet(cv_meta):
    return json.dumps({
        "name": CV_NAME,
        "uri": cv_meta.get("stable_uri") or OBO_RAW_URL,
        "version": cv_meta.get("data-version", "unknown"),
    }, indent=2)


def reconstruct_obo_block(raw_term):
    """reconstruct a faithful OBO block from a parsed term dict"""
    lines = ["[Term]", f"id: {raw_term['id']}", f"name: {raw_term['name']}"]
    if raw_term.get("def"):
        xrefs = ", ".join(raw_term.get("def_xrefs", ["PSI:MS"]))
        lines.append(f'def: "{raw_term["def"]}" [{xrefs}]')
    if raw_term.get("comment"):
        lines.append(f"comment: {raw_term['comment']}")
    for syn in raw_term.get("synonyms", []):
        #keep the original scope and xrefs instead of forcing RELATED []
        if isinstance(syn, dict):
            rest = syn.get("rest", "").strip()
            lines.append(
                f'synonym: "{syn["name"]}" {rest}'.rstrip()
                if rest else f'synonym: "{syn["name"]}" RELATED []'
            )
        else:
            lines.append(f'synonym: "{syn}" RELATED []')
    for x in raw_term.get("xrefs", []):
        lines.append(f"xref: {x}")
    for parent in raw_term.get("is_a", []):
        lines.append(f"is_a: {parent}")
    if raw_term.get("xsd_value_type"):
        lines.append(
            f"relationship: has_value_type {raw_term['xsd_value_type']}"
        )
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
    for po in raw_term.get("part_of", []):
        lines.append(f"relationship: part_of {po}")
    if raw_term.get("order"):
        lines.append(f"relationship: has_order {raw_term['order']}")
    if raw_term.get("domain"):
        lines.append(f"relationship: has_domain {raw_term['domain']}")
    if raw_term.get("is_obsolete"):
        lines.append("is_obsolete: true")
    if raw_term.get("replaced_by"):
        lines.append(f"replaced_by: {raw_term['replaced_by']}")
    return "\n".join(lines)


#accepted metrics persistence

def load_accepted_metrics():
    if os.path.exists(NEW_METRICS_FILE):
        try:
            with open(NEW_METRICS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", NEW_METRICS_FILE, e)
    return []


def save_accepted_metrics(accepted):
    try:
        with open(NEW_METRICS_FILE, "w") as f:
            json.dump(accepted, f, indent=2)
        return True
    except OSError as e:
        logger.warning("Could not write %s: %s", NEW_METRICS_FILE, e)
        return False


def append_new_metric(result, proposed_name, proposed_desc, cv_meta):
    accepted = load_accepted_metrics()
    #dedup on BOTH names, the same proposal re-analysed can come back under a slightly different drafted name
    existing_drafted = {
        (m.get("suggested_name") or "").lower() for m in accepted
    }
    existing_proposed = {
        (m.get("proposed_name") or "").lower() for m in accepted
    }
    if (result.get("suggested_name") or "").lower() in existing_drafted:
        return False, "already_saved"
    if proposed_name and proposed_name.lower() in existing_proposed:
        return False, "already_saved"
    record = {
        "proposed_name": proposed_name,
        "proposed_description": proposed_desc,
        "suggested_name": result.get("suggested_name"),
        "suggested_def": result.get("suggested_def"),
        "suggested_comment": result.get("suggested_comment"),
        "suggested_synonyms": result.get("suggested_synonyms"),
        "suggested_xrefs": result.get("suggested_xrefs"),
    }
    for dim_key in CLASSIFICATION_SCHEMA:
        record[dim_key] = result.get(dim_key)
    record["suggested_categories"] = result.get("suggested_categories")
    record["suggested_units"] = result.get("suggested_units")
    record["suggested_value_concept"] = result.get("suggested_value_concept")
    record["suggested_xsd_type"] = result.get("suggested_xsd_type")
    record["suggested_relations"] = result.get("suggested_relations")
    record["suggested_columns"] = result.get("suggested_columns")
    record["suggested_new_column_terms"] = result.get(
        "suggested_new_column_terms"
    )
    record["suggested_consolidation"] = result.get("suggested_consolidation")
    record["max_overlap_level"] = max_overlap_level(
        result.get("overlap_results", [])
    )
    record["overlap_results"] = result.get("overlap_results", [])
    record["verdict_summary"] = result.get("verdict_summary")
    #record the CV version the decision was made against, a saved metric cannot be reproduced once master has moved on
    record["cv_version"] = cv_meta.get("data-version")
    record["cv_uri"] = cv_meta.get("stable_uri")
    record["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    accepted.append(record)
    if not save_accepted_metrics(accepted):
        return False, "write_failed"
    return True, "saved"


def persist_if_new(result, cv_meta):
    """persistence happens ONCE right after an analysis, never inside the render path where every streamlit rerun would repeat it"""
    _, _, _, is_new_metric = compute_verdict_flags(result)
    if not (is_new_metric and result.get("suggested_name")):
        return None
    saved, status = append_new_metric(
        result,
        st.session_state.get("proposed_name", ""),
        st.session_state.get("proposed_desc", ""),
        cv_meta,
    )
    return status if not saved else "saved"


#prompt building

def build_classification_prompt_section():
    lines = [
        "Each QC metric must be classified along seven independent "
        "dimensions. For each dimension, choose exactly one value.",
        "",
    ]
    for i, (dim_key, dim_info) in enumerate(CLASSIFICATION_SCHEMA.items(), 1):
        lines.append(
            f"  Dimension {i} -- {dim_info['label']}"
            f' (JSON key: "{dim_key}"):'
        )
        lines.append(f"  {dim_info['description']}")
        for val_name, val_desc in dim_info["values"].items():
            lines.append(f'    - "{val_name}": {val_desc}')
        lines.append("")
    lines += [
        "CRITICAL DISAMBIGUATION RULES:",
        "",
        "  analytical_dimension -- decision rules:",
        "    MS2 ISOLATION WINDOW (narrow window around precursor for",
        "    fragmentation): isolation window width, isolation target m/z,",
        '    isolation boundaries, precursor purity -> "isolation purity metric".',
        "    MS1 SCAN WINDOW (full m/z range for survey scans): scan window",
        "    upper/lower m/z limit, configured acquisition range, scan counts,",
        '    sampling density, number of spectra -> "acquisition coverage metric".',
        "    KEY DISTINCTION: 'scan window' describes the full survey m/z",
        "    range (acquisition coverage). 'isolation window' describes the",
        "    narrow precursor selection window for MS2 (isolation purity).",
        "    These are DIFFERENT instrument settings -- do NOT confuse them.",
        "",
        "  workflow_stage vs data_dependency -- these are INDEPENDENT:",
        "    workflow_stage = where the MEASURED QUANTITY physically originates"
        " in the pipeline.",
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
        '    1. Purity, coverage, rate, completeness fraction where 1.0 = '
        'perfect -> "higher is better".',
        '    2. Error, deviation, contamination, bad event count, coefficient '
        'of variation -> "lower is better".',
        '    3. SIGNED deviation centered on a target (mass deviation where '
        '0 ppm is ideal, temperature) -> "target range".',
        '    4. "context dependent" ONLY when direction genuinely varies by '
        "experimental design.",
        "",
    ]
    return "\n".join(lines)


def _build_id_lookup_table(candidate_display_terms):
    """build a compact ID -> name reference table for the LLM"""
    rows = []
    for t in candidate_display_terms:
        rows.append(f"  {t['id']}  {t['name']}")
    return "\n".join(rows)


def _scope_hint_for(term_display):
    """scope hint for one candidate term, read from its categories only"""
    cats_lower = " ".join(term_display.get("categories", [])).lower()
    per_spectrum = "single spectrum" in cats_lower
    many_spectra = "multiple spectra" in cats_lower
    single_run = "single run" in cats_lower
    many_runs = "multiple runs" in cats_lower
    if per_spectrum:
        return "  (spectrum-level)"
    if many_runs:
        return "  (multi-run-level)"
    if single_run and many_spectra:
        return "  (run-level, aggregated over many spectra)"
    if single_run:
        return "  (run-level)"
    if many_spectra:
        return "  (aggregated over many spectra)"
    return ""


def _build_term_type_annotations(candidate_display_terms):
    """build a compact table labelling each candidate as base or QC metric"""
    rows = []
    for t in candidate_display_terms:
        if _is_qc_metric_term(t):
            kind = "QC_METRIC"
            scope_hint = _scope_hint_for(t)
        else:
            kind = "BASE_TERM"
            scope_hint = "  (per-occurrence attribute)"
        rows.append(f"  {t['id']}  [{kind}]{scope_hint}  {t['name']}")
    return "\n".join(rows)


def _build_vocab_tables(cv_index):
    cats = "\n".join(
        f"    {cid}  {cname}"
        for cname, cid in sorted(cv_index["category_name_to_id"].items())
    )
    units = "\n".join(
        f"    {uid}  {uname}"
        for uid, uname in sorted(
            cv_index["unit_id_to_name"].items(), key=lambda x: x[1].lower()
        )
    )
    concepts = "\n".join(
        f"    {cid}  {cname}"
        for cid, cname in sorted(
            cv_index["concept_id_to_name"].items(), key=lambda x: x[1].lower()
        )
    )
    vtypes = "\n".join(
        f"    {vid}  {vname}"
        for vname, vid in sorted(cv_index["vtype_name_to_id"].items())
    )
    return cats, units, concepts, vtypes


def _build_column_term_table(cv_index):
    return "\n".join(
        f"    {cid}  {cname}"
        for cid, cname in sorted(
            cv_index["used_column_id_to_name"].items(),
            key=lambda x: x[1].lower(),
        )
    )


def build_spec_rules_section(cv_index):
    """the hard rules from mzQC 1.0.0 section 4.1 (CV term requirements) plus section 7 (value types)"""
    cats, units, concepts, vtypes = _build_vocab_tables(cv_index)
    return "\n".join([
        "MZQC 1.0.0 SPECIFICATION RULES -- these are MUST/SHOULD rules from",
        "the standard, not style preferences. Violating a MUST makes the term",
        "unusable in an mzQC file.",
        "",
        "  NAME (spec 4.1):",
        "   - SHOULD be informative and at most 100 characters.",
        "   - SHOULD use only 7-bit ASCII alphanumerics, spaces and the",
        "     punctuation marks - _ , . (existing terms such as 'm/z",
        "     acquisition range' deviate, so m/z and S/N are tolerated).",
        "   - MUST NOT duplicate the name of an existing active term.",
        "",
        "  DEFINITION (spec 4.1):",
        "   - MUST explain purpose, requirements and scope of the metric and",
        "     how it is represented in an mzQC file.",
        "   - MUST NOT open with a filler clause such as 'Quality control",
        "     metric reporting ...' or 'A QC metric that describes ...'. Say",
        "     directly what the value is: 'The coefficient of variation of",
        "     quantified concentrations for each monitored feature ...'.",
        "   - SHOULD NOT contain the calculation or the interpretation, those",
        "     belong in the comment.",
        "   - MUST NOT contain double quotes, escaped characters or backticks.",
        "     Use single quotes to quote words.",
        "",
        "  COMMENT (spec 4.1, OPTIONAL):",
        "   - MAY carry calculation and interpretation details: whether higher",
        "     or lower is better, division by zero and empty-run edge cases,",
        "     dependence on the instrument method.",
        "   - MUST NOT merely repeat name, definition or synonyms.",
        "",
        "  SYNONYMS (spec 4.1, OPTIONAL): scope is EXACT (pure rename),",
        "  NARROW (same interpretation, different calculation) or RELATED",
        "  (similar but not the same). Give the publication or software name",
        "  of the metric as a synonym with a PMID xref when it is published,",
        "  e.g. synonym: 'XIC-WideFrac' EXACT [PMID:24494671]. Prefer a",
        "  synonym over minting a new term for a tool-specific name.",
        "",
        "  XREFS (spec 4.1, OPTIONAL): suggested_xrefs carries external",
        "  references that define the concept, as PMID:12345678, DOI:10.x/y",
        "  or a documentation URL. Use this when the concept is defined by an",
        "  external resource rather than by the CV. Never invent a PMID or DOI.",
        "",
        "  VALUE TYPE (spec 4.1 and section 7): every metric MUST declare one",
        "  value type through is_a. Available in this CV version:",
        vtypes,
        "   - single value: one number, string or boolean. has_units REQUIRED.",
        "   - n-tuple: JSON array, uniform type AND uniform unit.",
        "     has_units REQUIRED.",
        "   - table: JSON object of equal-length columns. has_column REQUIRED",
        "     (one or more), has_optional_column OPTIONAL. The table itself",
        "     MUST NOT declare has_units, each column term carries its own.",
        "   - matrix: array of equal-length arrays, uniform type and unit.",
        "     has_units REQUIRED.",
        "",
        "  UNITS (spec 4.1): pick exactly ONE unit for single value, n-tuple",
        "  and matrix metrics, from this list only. If the quantity is truly",
        "  dimensionless (entropy, score, index) use the dimensionless unit",
        "  rather than omitting the unit, because a unit is REQUIRED:",
        units,
        "",
        "  VALUE CONCEPT (spec 4.1, RECOMMENDED): the statistical concept the",
        "  value represents. Use ONLY these accessions:",
        concepts,
        "",
        "  METRIC CATEGORIES (spec 4.1): it is RECOMMENDED to state ID based",
        "  vs ID free, plus context categories. Use ONLY these accessions:",
        cats,
        "",
        "  TABLE COLUMN TERMS: these accessions are already used as columns",
        "  in this CV and are ALWAYS allowed in suggested_columns, in addition",
        "  to anything in the id_lookup_table. Some of them come from other",
        "  ontologies, that is expected and correct:",
        _build_column_term_table(cv_index),
        "",
        "  XSD VALUE TYPE (RECOMMENDED): one of "
        + ", ".join(ALLOWED_XSD_TYPES) + ", but the QC namespace only uses "
        + ", ".join(QC_PREFERRED_XSD_TYPES)
        + ". Use xsd:int for counts and xsd:float for continuous quantities.",
        "",
    ])


def build_system_prompt(candidate_display_terms, total_count, active_count,
                        cv_index, cv_meta):
    metrics_text = json.dumps(candidate_display_terms, indent=1)
    classification_section = build_classification_prompt_section()
    spec_section = build_spec_rules_section(cv_index)
    dim_keys_example = ",\n".join(
        f'  "{k}": "..."' for k in CLASSIFICATION_SCHEMA
    )
    non_metric_parent = (
        f"{cv_index['non_metric_parent_id']} ! "
        f"{cv_index['non_metric_parent_name']}"
    )

    id_lookup = _build_id_lookup_table(candidate_display_terms)
    term_type_table = _build_term_type_annotations(candidate_display_terms)

    json_example = (
        "{\n"
        '  "needs_more_detail": false,\n'
        '  "clarification_question": null,\n'
        '  "suggested_name": "metric name",\n'
        '  "suggested_def": "OBO definition text.",\n'
        '  "suggested_comment": "Interpretation guidance or null.",\n'
        '  "suggested_synonyms": [\n'
        '    {"name": "published-name", "scope": "EXACT", '
        '"xref": "PMID:24494671"}\n'
        "  ],\n"
        '  "suggested_xrefs": ["PMID:24494671"],\n'
        "\n"
        + dim_keys_example + ",\n"
        "\n"
        '  "suggested_categories": ["ID free metric", "MS2 metric"],\n'
        '  "suggested_units": ["parts per million"],\n'
        '  "suggested_value_concept": '
        '{"id": "STATO:0000401", "name": "sample mean"},\n'
        '  "suggested_xsd_type": "xsd:float",\n'
        '  "suggested_relations": [\n'
        '    {"id": "MS:4000072", "name": "observed mass accuracy"}\n'
        "  ],\n"
        '  "suggested_columns": [\n'
        '    {"id": "MS:1000041", "name": "charge state", "optional": false}\n'
        "  ],\n"
        '  "suggested_new_column_terms": [\n'
        "    {\n"
        '      "name": "signal-to-blank ratio",\n'
        '      "def": "Ratio between the quantitative signal measured in a '
        'sample and the quantitative signal measured in a corresponding blank '
        'sample.",\n'
        '      "xsd_type": "xsd:float",\n'
        '      "unit": "ratio",\n'
        '      "value_concept": null,\n'
        '      "optional": false,\n'
        '      "xrefs": []\n'
        "    }\n"
        "  ],\n"
        '  "suggested_consolidation": null,\n'
        "\n"
        '  "overlap_results": [\n'
        "    {\n"
        '      "id": "MS:4000026",\n'
        '      "name": "fragment ppm deviation median",\n'
        '      "overlap_level": "moderate",\n'
        '      "is_obsolete": false,\n'
        '      "reasoning": "Both measure mass accuracy deviation but at '
        'different MS levels."\n'
        "    }\n"
        "  ],\n"
        '  "is_new": true,\n'
        '  "verdict_summary": "This metric is new."\n'
        "}"
    )

    prompt_parts = [
        "You are an expert in mass spectrometry quality control metrics, "
        "specifically the PSI-MS controlled vocabulary used in mzQC files, "
        "and you apply the mzQC 1.0.0 specification strictly.\n",

        f"The vocabulary below was downloaded from the psi-ms-CV repository "
        f"(data-version {cv_meta.get('data-version', 'unknown')}).\n",

        "Below is a CURATED SUBSET of the most relevant existing terms, "
        f"retrieved from the FULL psi-ms.obo ontology ({total_count} terms "
        f"total, {active_count} active). The subset includes both QC metric "
        "terms and the base MS vocabulary terms they reference "
        "(instrument concepts, spectrum types, scan attributes, etc.).\n\n"
        "Each entry may have:\n"
        "  - id, name, def, comment, synonyms, xrefs\n"
        "  - is_a: parent term(s)\n"
        "  - part_of: compositional parent(s)\n"
        '  - value_type: "single value", "n-tuple", "table", "matrix"\n'
        "  - categories: OBO metric categories\n"
        "  - units: measurement units\n"
        "  - value_concepts: statistical concepts\n"
        "  - relations: list of related CV terms via has_relation\n"
        "  - columns / optional_columns: for table-type metrics\n"
        "  - order: ordering hint\n"
        "  - domain: value domain constraint\n"
        "  - is_obsolete, replaced_by\n",

        "<relevant_terms>\n" + metrics_text + "\n</relevant_terms>\n",

        #id accuracy rules
        "CRITICAL -- ACCESSION NUMBER ACCURACY RULES:\n"
        "The following is the COMPLETE list of IDs available to you for\n"
        "overlap_results and suggested_relations.\n"
        "You MUST NOT fabricate, guess, or recall from memory any accession\n"
        "number that is not in this list. Every id you write in overlap_results,\n"
        "suggested_relations, or the definition text MUST appear here.\n"
        "Units, categories, value concepts and table columns are the\n"
        "exception: for those use the dedicated specification lists further\n"
        "down.\n"
        "If you believe a relevant term exists but is not in this list, say so\n"
        "in verdict_summary WITHOUT inventing an accession number.\n\n"
        "<id_lookup_table>\n" + id_lookup + "\n</id_lookup_table>\n\n",

        #term type annotations
        "CRITICAL -- TERM TYPE CLASSIFICATION:\n"
        "Each candidate term is classified below as either BASE_TERM or\n"
        "QC_METRIC, with a scope hint. You MUST use this information during\n"
        "overlap analysis (Step 8). BASE_TERM entries define raw attributes,\n"
        "instrument concepts, or per-occurrence quantities. QC_METRIC entries\n"
        "define aggregated quality measures.\n\n"
        "<term_type_table>\n" + term_type_table + "\n</term_type_table>\n\n",

        f"NOTE: {len(candidate_display_terms)} of {total_count} terms shown "
        "(pre-filtered by semantic similarity and graph neighbors). If you "
        "believe a relevant term might exist outside this subset, mention it "
        "in your verdict_summary but do NOT fabricate an ID for it.\n",

        "<specification_rules>\n" + spec_section + "</specification_rules>\n",

        "A user is proposing a new QC metric. Perform the following steps.\n",

        "STEP 1 -- ASSESS DESCRIPTION QUALITY:\n"
        "It must specify: (a) what quantity is measured, (b) how it is "
        "computed, (c) what MS level, (d) ID-based or ID-free, "
        "(e) unit/value type.\n"
        "If any are missing, set needs_more_detail to true and write a "
        "specific clarification_question. Still provide best attempt at all "
        "other fields.\n",

        #step 2: naming with abbreviation casing rules
        "STEP 2 -- GENERATE FORMAL NAME:\n"
        "Short, precise, lowercase EXCEPT for standard mass spectrometry "
        "abbreviations which MUST stay uppercase. Follow patterns: "
        '"X distribution mean", "number of X", "X quantiles", etc. '
        "WORD ORDER: the CV puts the quantity first and the statistic LAST, "
        'so "MS2 precursor intensity median", never "median MS2 precursor '
        'intensity". Keep the user\'s word order when it already follows '
        "this convention. "
        "Respect the name rules in <specification_rules>. No accession in the "
        "name.\n\n"
        "FAITHFULNESS RULE -- this governs the WHOLE response:\n"
        "You are evaluating the proposal as written. suggested_name,\n"
        "metric_value_type, measurement_scope and the OBO fields MUST\n"
        "describe THAT metric, at the scope and value type the user\n"
        "described. If the user proposes a single value, draft a single\n"
        "value term, even when you believe a table of several parameters\n"
        "would be the better design. If you would name the term something\n"
        "substantially different from the proposed name, that is a signal\n"
        "that you have redesigned the metric rather than drafted it.\n"
        "Every redesign proposal belongs in suggested_consolidation (Step\n"
        "8b) and NOWHERE else. Tightening the wording of the name is fine,\n"
        "changing what the term measures is not.\n\n"
        "ABBREVIATION CASING RULES -- these MUST be uppercase in the name:\n"
        "  MS1, MS2, MS3, MSn, DDA, DIA, SWATH, SRM, PRM, MRM,\n"
        "  TIC, XIC, BPC, FWHM, FDR, PSM, CCS, IMS, TOF,\n"
        "  ESI, MALDI, DESI, LC, GC, CE, RT, S/N, BQC, TQC, LOD, LOQ.\n"
        "Examples of correct casing:\n"
        '  "MS2 spectral entropy mean" (not "ms2 spectral entropy mean")\n'
        '  "DIA isolation window count" (not "dia isolation window count")\n',

        #step 3: definition + comment, split exactly as the spec demands
        "STEP 3 -- GENERATE FORMAL DEFINITION, COMMENT, SYNONYMS AND XREFS:\n"
        "suggested_def: purpose, requirements and scope of the metric plus how\n"
        "it is stored in an mzQC file. Start with the quantity itself, NEVER\n"
        "with 'Quality control metric reporting' or any equivalent filler.\n"
        "Do NOT put the calculation recipe or the good/bad interpretation\n"
        "here. No double quotes, no backticks.\n"
        "When referencing other MS terms by accession, use ONLY accessions\n"
        "from the id_lookup_table above.\n\n"
        "suggested_comment (may be null): the non-trivial information that does\n"
        "NOT belong in the definition:\n"
        "  - Whether higher or lower values are desirable and why.\n"
        "  - Edge cases (division by zero, empty runs, single-width runs).\n"
        "  - How the metric relates to instrument configuration.\n"
        "  - A brief description of the calculation if it is not obvious.\n"
        "Do NOT repeat the definition or name in the comment.\n\n"
        "suggested_synonyms (may be an empty list): if the proposal restates a\n"
        "published or software-specific metric, add that name with the right\n"
        "scope (EXACT/NARROW/RELATED) and a PMID xref if you are certain of\n"
        "it, otherwise leave the xref empty. Never invent a PMID.\n\n"
        "suggested_xrefs (may be an empty list): external references that\n"
        "define the underlying concept, such as a PMID, a DOI or a\n"
        "documentation URL. Use this when the user cites an external\n"
        "definition for terminology the CV does not define itself.\n",

        "STEP 4 -- SEVEN-DIMENSION CLASSIFICATION:\n"
        + classification_section,

        "STEP 5 -- OBO VALUE FIELDS (follow <specification_rules> exactly):\n"
        "  - suggested_categories: names from the category list. ALWAYS\n"
        "    include exactly one of 'ID based metric' or 'ID free metric':\n"
        "    ID based if the metric needs identified peptides, compounds or\n"
        "    lipids (including as a filter or as the row key), ID free if it\n"
        "    can be computed from raw or quantitative data alone.\n"
        "    * A table with one row per spectrum reports on many spectra of\n"
        "      one run, so it SHOULD carry both 'single run based metric' and\n"
        "      'multiple spectra based metric'.\n"
        "    * A table with one row per feature whose values summarise several\n"
        "      runs (a CV across replicate injections, a regression over a\n"
        "      dilution series, a ratio of sample to blank runs) is a\n"
        "      MULTI-RUN metric and MUST carry 'multiple runs based metric',\n"
        "      not 'single run based metric'.\n"
        "    * Anything computed from concentrations, areas or normalised\n"
        "      intensities is a 'quantification based metric'.\n"
        "  - suggested_units: exactly ONE unit name from the unit list for\n"
        "    single value / n-tuple / matrix, and an EMPTY list for a table.\n"
        "  - suggested_value_concept: one entry from the value concept list,\n"
        "    or null if none fits.\n"
        "  - suggested_xsd_type: one of the allowed xsd types.\n",

        "STEP 6 -- RELATIONSHIPS (has_relation):\n"
        "Identify existing CV terms for the has_relation field. Be SELECTIVE\n"
        "-- only include terms that have a DIRECT semantic link. The goal is\n"
        "a small, stable set of the most relevant relations, not an\n"
        "exhaustive list of everything tangentially related.\n\n"

        "INCLUDE in suggested_relations (MUST or SHOULD):\n"
        "  MUST: The base term whose quantity the metric directly aggregates\n"
        "        or summarizes (e.g. 'scan window upper limit' for a metric\n"
        "        that reports the MS1 scan window upper bound).\n"
        "  MUST: Any term referenced by accession in the definition text.\n"
        "  SHOULD: The single closest companion QC metric that measures the\n"
        "          same property in a complementary way (e.g. the existing\n"
        "          DIA-specific version of a mode-independent metric, or\n"
        "          the lower-limit companion of an upper-limit metric).\n\n"

        "EXCLUDE from suggested_relations (do NOT include):\n"
        "  - The metric's own has_metric_category terms. A term that belongs\n"
        "    in suggested_categories (e.g. 'MS2 metric', 'ID free metric')\n"
        "    MUST NOT also appear in suggested_relations -- an accession is\n"
        "    either a category or a relation, never both.\n"
        "  - Terms used as has_column, has_units or has_value_concept for this\n"
        "    same metric, for the same reason.\n"
        "  - Several near-synonymous terms for the same concept. If the CV\n"
        "    contains three variants of 'coefficient of variation', pick the\n"
        "    ONE that fits best and leave the others out.\n"
        "  - Obsolete terms (is_obsolete: true) unless no active alternative\n"
        "    exists and the obsolete term is the only direct match.\n"
        "  - Vendor-specific processing parameters (e.g. ProteomeDiscoverer,\n"
        "    MaxQuant-specific terms) unless the proposed metric is itself\n"
        "    vendor-specific.\n"
        "  - Terms that only share a unit or category with the proposal.\n"
        "  - Terms from the overlap_results at 'low' level -- if a term is\n"
        "    only loosely related, it is not a has_relation candidate.\n"
        "  - Terms that describe a DIFFERENT property of the same instrument\n"
        "    component (e.g. 'isolation window upper offset' is NOT a\n"
        "    relation for 'scan window upper limit' -- different property,\n"
        "    different instrument component).\n\n"

        "TARGET: 1-4 relations for a typical metric. Fewer is better than\n"
        "more. If you are unsure whether to include a term, leave it out.\n"
        'Report as suggested_relations: a list of objects with "id" and\n'
        '"name". ONLY use IDs from the id_lookup_table.\n',

        'STEP 7 -- TABLE COLUMNS (if metric_value_type is "table"):\n'
        "A table MUST have at least one has_column, a table without columns is\n"
        "invalid and cannot be used in an mzQC file.\n\n"
        "  (a) EXISTING columns go in suggested_columns as objects with id,\n"
        "      name and optional (true for has_optional_column). Accessions\n"
        "      may come from EITHER the id_lookup_table OR the TABLE COLUMN\n"
        "      TERMS list in <specification_rules>. That list is an explicit\n"
        "      exception to the accession accuracy rule: those ids are\n"
        "      verified and you SHOULD use them, several are from other\n"
        "      ontologies.\n"
        "  (b) If a genuinely required column does NOT exist in either list,\n"
        "      do NOT invent an accession and do NOT silently drop the column.\n"
        "      Put it in suggested_new_column_terms with name, def, xsd_type,\n"
        "      unit (a name from the unit list, or null for an identifier or\n"
        "      label column), optional and xrefs. The app drafts each of these\n"
        f"      as its own OBO term under {non_metric_parent}, which is where\n"
        "      column terms belong: they are not metrics, and a term with no\n"
        "      is_a floats at the top level of the CV.\n"
        "      Keep these column terms GENERIC and reusable across studies,\n"
        "      not specific to one workflow or one software package.\n"
        "      Each entry may also carry value_concept: the statistical\n"
        "      concept the column reports, chosen from the VALUE CONCEPT\n"
        "      list in <specification_rules>. A regression parameter, a\n"
        "      coefficient of variation, a mean or a median SHOULD carry\n"
        "      one. If the list holds nothing suitable, leave it null and\n"
        "      say in verdict_summary which concept would be needed, in\n"
        "      words, with no invented accession.\n"
        "      IDENTIFIER COLUMNS MUST STATE THEIR FORMAT. A definition\n"
        "      such as the identifier of the feature in this row is NOT\n"
        "      acceptable: it tells a reader nothing about what to put in\n"
        "      the column and makes the term unusable across tools. Say\n"
        "      what identifies the row: an accurate mass and retention\n"
        "      time pair, a transition or precursor-to-product m/z pair, a\n"
        "      database or spectral library accession, or a specified\n"
        "      textual name. A feature-level equivalent of the USI does\n"
        "      not exist yet, so note in verdict_summary that the format\n"
        "      has to be stated explicitly until one does.\n"
        "  (c) NEVER substitute a near-neighbour. A column accession must\n"
        "      match the MEANING of the column, not merely its role or its\n"
        "      data type. If the table needs a generic feature identifier,\n"
        "      do NOT reach for a peptide-specific term such as a proforma\n"
        "      peptidoform sequence, and if it needs a regression parameter,\n"
        "      do NOT reach for a term that merely labels the quantification\n"
        "      datatype. A drafted new column term is ALWAYS better than a\n"
        "      semantically wrong existing accession: the draft is reviewed\n"
        "      before it enters the CV, the wrong accession is not.\n"
        "      Use an existing peptide-specific or spectrum-specific column\n"
        "      ONLY when the metric itself is peptide-centric or\n"
        "      spectrum-centric. Lipidomics, metabolomics and other\n"
        "      feature-centric metrics need a generic feature identifier.\n"
        "  (d) A column is EITHER the identifier of the row OR a quantity the\n"
        "      table actually reports for every row. A term that labels the\n"
        "      workflow rather than carrying a per-row value is NOT a column:\n"
        "      a quantification datatype, a software setting or an\n"
        "      acquisition parameter is constant for the whole table, so it\n"
        "      belongs in the definition or in suggested_relations. Adding it\n"
        "      as a column forces a null into the mzQC unit array and gives\n"
        "      the reader a column with nothing to put in it.\n"
        "  (e) A term used as a column MUST NOT also appear in\n"
        "      suggested_relations, and vice versa. Decide the role once.\n"
        "  A per-spectrum table SHOULD carry a spectrum identifier column such\n"
        "  as the native spectrum identifier format, plus one column per\n"
        "  reported quantity. A per-feature table SHOULD carry a feature\n"
        "  identifier column plus one column per reported quantity.\n"
        "  For non-table metrics return empty lists for both fields.\n",

        #overlap analysis (with precise borderline rules)

        "STEP 8 -- OVERLAP ANALYSIS (BE THOROUGH, PRECISE, AND "
        "SCOPE-AWARE):\n"
        "Compare the proposed QC metric against the provided terms.\n\n"

        "FUNDAMENTAL DISTINCTION -- BASE TERMS vs QC METRICS:\n"
        "  (a) BASE TERMS (BASE_TERM in term_type_table): raw per-occurrence\n"
        "      attributes. NEVER 'duplicate' or 'high' vs a QC metric.\n"
        "  (b) QC METRICS (QC_METRIC in term_type_table): aggregated quality\n"
        "      measures.\n\n"

        "OVERLAP LEVEL DEFINITIONS:\n"
        '  "duplicate" = SAME quantity, SAME statistic, SAME scope, SAME\n'
        "               filtering, SAME value type, SAME acquisition mode\n"
        "               applicability. Both must be QC metrics.\n"
        '  "high"      = SAME quantity AND same scope, but output is derivable\n'
        "               from an existing QC metric (e.g. median from quantile\n"
        "               tuple). Both must be QC metrics. ALSO requires same\n"
        "               acquisition mode applicability (see below).\n"
        '  "moderate"  = Related quantity at same MS level and scope (mean vs\n'
        "               sigma), OR same quantity at DIFFERENT scopes, OR a base\n"
        "               term that defines the aggregated quantity.\n"
        '  "low"       = Loosely related concept.\n\n'

        "ACQUISITION MODE SPECIFICITY RULES (critical for consistent verdicts):\n"
        "  When comparing two terms, check whether one is mode-specific\n"
        "  (e.g. DIA-only, DDA-only) and the other is mode-independent.\n"
        "  - If the existing term is restricted to a specific acquisition mode\n"
        "    (e.g. 'DIA isolation window m/z widths' is DIA-only) but the\n"
        "    proposed metric applies to ALL acquisition modes, they are NOT\n"
        "    duplicates. The proposed metric has BROADER applicability.\n"
        '    Maximum overlap = "moderate".\n'
        "  - If both are mode-independent OR both are the same mode: normal\n"
        "    overlap rules apply.\n"
        "  - ALWAYS note acquisition mode differences in the reasoning.\n\n"

        "VALUE TYPE DIFFERENCE RULES:\n"
        "  When comparing two terms, check their value types (single value,\n"
        "  n-tuple, table, matrix).\n"
        "  - If the existing term is an n-tuple (e.g. min/max pair) and the\n"
        "    proposed metric is a single value (e.g. median), they report\n"
        "    DIFFERENT statistics even if the underlying quantity is the same.\n"
        '    This alone caps overlap at "moderate".\n'
        "  - Only rate as 'high' when the proposed single value is directly\n"
        "    contained in or extractable from the existing term's output\n"
        "    (e.g. Q2 from a quantile tuple). A median is NOT extractable\n"
        "    from a min/max pair.\n\n"

        "SCOPE-AWARENESS RULES:\n"
        "  - ALWAYS compare measurement scope between proposed and existing.\n"
        "  - Different scopes = at most 'moderate' overlap.\n"
        "  - Scope difference must always be mentioned in reasoning.\n\n"

        "CONSISTENCY RULES:\n"
        '  - Mean vs sigma of same distribution at same scope = "moderate".\n'
        "  - Single value extractable from existing quantile tuple at same\n"
        '    scope AND same mode = "high".\n'
        "  - Sharing a unit or category alone is NOT enough for overlap.\n"
        "  - Base term directly aggregated = suggested_relations + at most\n"
        '    "moderate" overlap.\n'
        "  - When in doubt about scope, check: does the existing term produce\n"
        "    one value per spectrum/scan, or one value per run?\n"
        "  - Also consider companion QC metrics as low-overlap or relation\n"
        "    candidates.\n\n"

        "CHECKLIST (apply to EACH candidate before assigning a level):\n"
        "  1. Is the candidate a BASE_TERM or QC_METRIC?\n"
        "  2. Same underlying quantity?\n"
        "  3. Same statistic (mean, median, count, min/max, quantiles)?\n"
        "  4. Same measurement scope (spectrum, run, batch)?\n"
        "  5. Same acquisition mode applicability (DIA-only vs mode-independent)?\n"
        "  6. Same value type (single value vs n-tuple vs table)?\n"
        "  7. Is the proposed value extractable from the existing output?\n"
        "  Only if ALL of 1-7 match can the overlap be 'duplicate'.\n"
        "  If 1-4 match but 5 or 6 differ, cap at 'moderate'.\n"
        "  If 1-4 match and 7 is true and 5-6 match, 'high'.\n\n"

        "For each overlapping term report: id, name, overlap_level, "
        "is_obsolete, reasoning.\n"
        "IMPORTANT: The id and name MUST exactly match the id_lookup_table.\n"
        "IMPORTANT: In reasoning, ALWAYS state the scope, acquisition mode,\n"
        "value type, and statistic of BOTH terms.\n",

        "STEP 8b -- CONSOLIDATION CHECK:\n"
        "Separately from overlap, ask whether the proposal is one of SEVERAL\n"
        "parameters that all fall out of the SAME computation, at the same\n"
        "scope, over the same input. Classic case: the slope, the intercept\n"
        "and the coefficient of determination of one linear regression, or the\n"
        "mean and the standard deviation of one distribution reported per\n"
        "feature. Such parameters SHOULD be one table metric with one column\n"
        "per parameter rather than several single-value metrics, because they\n"
        "are always produced and interpreted together.\n"
        "If that applies, set suggested_consolidation to a short paragraph\n"
        "naming the sibling parameters and the combined table term you would\n"
        "define instead (name plus the column list). Otherwise set it to null.\n"
        "This does NOT change is_new or the overlap levels, and it MUST NOT\n"
        "change the drafted term: suggested_name, metric_value_type,\n"
        "measurement_scope, suggested_columns and suggested_units still\n"
        "describe the metric the user proposed, per the faithfulness rule in\n"
        "Step 2. Writing the consolidated table INTO those fields instead of\n"
        "into suggested_consolidation silently answers a question the user\n"
        "did not ask.\n"
        "If the proposal ALREADY has the consolidated shape, say so in one\n"
        "sentence rather than restating its columns.\n",

        "STEP 9 -- VERDICT:\n"
        '  - Any "duplicate" (among QC metrics): metric IS a duplicate.\n'
        '  - Highest "high" (among QC metrics): flag HIGH OVERLAP.\n'
        "  - Otherwise: metric is NEW.\n"
        "  A 'high' or 'duplicate' verdict is ONLY valid when the overlapping\n"
        "  term is a QC metric, NOT a base term.\n",

        "RESPONSE FORMAT -- ONLY valid JSON, no markdown fences:\n\n"
        + json_example + "\n\n"
        "Use the EXACT string values for each dimension. Do not invent new "
        "values.",
    ]

    return "\n".join(prompt_parts)


#LLM call

def call_gpt(messages, api_key):
    client = OpenAI(api_key=api_key, timeout=API_TIMEOUT)
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "temperature": 0,   
        "seed": 42,
    }
    try:
        response = client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        logger.warning("Retrying without temperature/seed: %s", e)
        kwargs.pop("temperature", None)
        kwargs.pop("seed", None)
        try:
            response = client.chat.completions.create(**kwargs)
        except APITimeoutError:
            return None, "The API request timed out. Please try again."
        except APIError as e2:
            return None, f"OpenAI API error: {e2}"
    except APITimeoutError:
        return None, "The API request timed out. Please try again."
    except APIError as e:
        return None, f"OpenAI API error: {e}"

    if not response.choices:
        return None, "The model returned no choices."
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    #message.content is None on a refusal or a hard token cut-off
    raw = (getattr(choice.message, "content", None) or "").strip()
    if not raw:
        if finish_reason == "length":
            return None, (
                "The model hit the token limit before returning any content. "
                f"Raise MAX_COMPLETION_TOKENS (currently "
                f"{MAX_COMPLETION_TOKENS}) or shorten the proposal."
            )
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            return None, f"The model declined to answer: {refusal}"
        return None, (
            "The model returned an empty response "
            f"(finish_reason={finish_reason})."
        )
    if finish_reason == "length":
        logger.warning("Completion was truncated, JSON parsing may fail")
    return _parse_json_response(raw)


def _find_json_object(text):
    """scan for the first balanced {...} block, ignoring braces inside string literals where a single '{' would throw the depth counter off"""
    depth, start = 0, None
    in_string, escaped = False, False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : i + 1]
                    start = None


def _parse_json_response(raw):
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass
    for candidate in _find_json_object(cleaned):
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            continue
    return None, (
        "Could not parse the model response as JSON.\n\n"
        f"Raw response (first 1500 chars):\n{raw[:1500]}"
    )


#post processing

def _validate_returned_ids(result, valid_id_set, cv_index):
    """strip accession numbers the model fabricated. overlaps plus relations must come from the candidate set. columns plus value concepts may be any CV term because retrieval does not surface every legal column"""
    warnings = []
    cleaned_overlaps = []
    for item in result.get("overlap_results", []):
        tid = item.get("id", "")
        if tid in valid_id_set:
            cleaned_overlaps.append(item)
        else:
            warnings.append(
                f"Removed overlap entry with unverified ID {tid} "
                f"(\"{item.get('name', '?')}\")."
            )
    result["overlap_results"] = cleaned_overlaps

    cleaned_rels = []
    for rel in result.get("suggested_relations", []):
        rid = rel.get("id", "")
        if rid in valid_id_set:
            cleaned_rels.append(rel)
        else:
            warnings.append(
                f"Removed suggested_relation with unverified ID {rid} "
                f"(\"{rel.get('name', '?')}\")."
            )
    result["suggested_relations"] = cleaned_rels

    cleaned_cols = []
    promoted = []
    for col in result.get("suggested_columns") or []:
        cid, cname = resolve_column(cv_index, col)
        if cid:
            cleaned_cols.append({
                "id": cid, "name": cname,
                "optional": bool(col.get("optional"))
                if isinstance(col, dict) else False,
            })
        else:
            label = col.get("name") if isinstance(col, dict) else str(col)
            if label:
                promoted.append({
                    "name": label,
                    "def": col.get("def") if isinstance(col, dict) else "",
                    "xsd_type": (
                        col.get("xsd_type") if isinstance(col, dict) else ""
                    ),
                    "unit": col.get("unit") if isinstance(col, dict) else None,
                    "optional": (
                        bool(col.get("optional"))
                        if isinstance(col, dict) else False
                    ),
                    "xrefs": [],
                })
                warnings.append(
                    f"Column '{label}' is not in the CV, it was moved to the "
                    "proposed new column terms."
                )
    result["suggested_columns"] = cleaned_cols

    #a proposed column whose name already exists in cv is not new, resolve it into a real column rather than drafting a duplicate term
    kept_new = []
    for nc in (result.get("suggested_new_column_terms") or []) + promoted:
        if not isinstance(nc, dict):
            nc = {"name": str(nc)}
        name = (nc.get("name") or "").strip()
        if not name:
            continue
        cid, cname = resolve_column(cv_index, {"id": "", "name": name})
        if cid:
            if cid not in {c["id"] for c in cleaned_cols}:
                cleaned_cols.append({
                    "id": cid, "name": cname,
                    "optional": bool(nc.get("optional")),
                })
            warnings.append(
                f"Proposed column term '{name}' already exists as {cid}, it "
                "was used directly instead of being drafted as a new term."
            )
            continue
        kept_new.append(nc)
    result["suggested_columns"] = cleaned_cols
    result["suggested_new_column_terms"] = kept_new

    concept = result.get("suggested_value_concept")
    if concept:
        cid, cname = resolve_value_concept(cv_index, concept)
        if cid:
            result["suggested_value_concept"] = {"id": cid, "name": cname}
        else:
            warnings.append(
                "Removed value concept "
                f"{concept if isinstance(concept, str) else concept.get('id')}"
                " which is not a CV term."
            )
            result["suggested_value_concept"] = None

    if warnings:
        existing_verdict = result.get("verdict_summary", "")
        note = (
            " [Post-processing note: "
            + " ".join(warnings)
            + "]"
        )
        result["verdict_summary"] = existing_verdict + note
    return result


def _enforce_base_term_overlap_cap(result, candidate_display_terms):
    """downgrade duplicate/high overlap with a base term to moderate"""
    base_term_ids = {
        t["id"] for t in candidate_display_terms
        if not _is_qc_metric_term(t)
    }
    if not base_term_ids:
        return result

    relation_ids = {
        r.get("id") for r in result.get("suggested_relations", [])
    }

    downgraded = []
    for item in result.get("overlap_results", []):
        tid = item.get("id", "")
        level = item.get("overlap_level", "low").lower()
        if tid not in base_term_ids:
            continue
        if level in ("duplicate", "high"):
            item["overlap_level"] = "moderate"
            item["reasoning"] = (
                f"[Auto-corrected from '{level}' to 'moderate': "
                f"{item.get('name', tid)} is a base vocabulary term "
                f"(per-occurrence attribute), not a QC metric.] "
                + item.get("reasoning", "")
            )
            downgraded.append(item)
            level = "moderate"
        if level == "moderate" and tid not in relation_ids:
            result.setdefault("suggested_relations", []).append({
                "id": tid, "name": item.get("name", ""),
            })
            relation_ids.add(tid)

    if downgraded:
        new_top = max_overlap_level(result.get("overlap_results", []))
        if new_top not in ("duplicate", "high"):
            result["is_new"] = True
            names = ", ".join(
                f"{d.get('id')} ({d.get('name', '?')})" for d in downgraded
            )
            existing_verdict = result.get("verdict_summary", "")
            result["verdict_summary"] = (
                f"[Auto-corrected verdict: overlap with base term(s) "
                f"{names} was downgraded to 'moderate'.] "
                + existing_verdict
            )
    return result


def _fix_result_name_casing(result):
    """deterministic fix for abbreviation casing in suggested name"""
    if result.get("suggested_name"):
        result["suggested_name"] = _fix_name_casing(result["suggested_name"])
    return result


def _enforce_value_type_rules(result, cv_index):
    """deterministic enforcement of the spec 4.1 value rules: a table never keeps units, a non table never keeps columns, a non table keeps at most one resolvable unit"""
    vtype = normalize_value_type(result.get("metric_value_type"))
    result["metric_value_type"] = vtype
    notes = []

    if vtype == "table":
        if result.get("suggested_units"):
            notes.append(
                "units were dropped because a table takes its units from its "
                "column terms"
            )
            result["suggested_units"] = []
    else:
        if result.get("suggested_columns"):
            notes.append(
                f"table columns were dropped because the value type is "
                f"'{vtype}'"
            )
            result["suggested_columns"] = []
        if result.get("suggested_new_column_terms"):
            notes.append(
                f"proposed new column terms were dropped because the value "
                f"type is '{vtype}'"
            )
            result["suggested_new_column_terms"] = []
        resolved, seen = [], set()
        for u in result.get("suggested_units") or []:
            uid, uname = resolve_unit(cv_index, u)
            if uid and uid not in seen:
                seen.add(uid)
                resolved.append(uname)
        if len(resolved) > 1:
            notes.append(
                f"kept only the first unit '{resolved[0]}', a "
                f"{vtype} must have one uniform unit"
            )
            resolved = resolved[:1]
        if resolved:
            result["suggested_units"] = resolved

    if notes:
        result["verdict_summary"] = (
            result.get("verdict_summary", "")
            + " [Spec enforcement: " + "; ".join(notes) + ".]"
        )
    return result


def _stabilize_relations(result, candidate_display_terms, cv_index):
    """keep relations small, stable, directly relevant. drops obsolete terms, vendor terms, low overlap terms, accessions already used in another role"""
    rels = result.get("suggested_relations", [])
    if not rels:
        return result

    id_to_term = {t["id"]: t for t in candidate_display_terms}

    overlap_levels = {}
    for item in result.get("overlap_results", []):
        tid = item.get("id", "")
        lvl = item.get("overlap_level", "low").lower()
        overlap_levels[tid] = lvl

    used_ids = set()
    for c in result.get("suggested_categories") or []:
        cid, _ = resolve_category(cv_index, c)
        if cid:
            used_ids.add(cid)
    for u in result.get("suggested_units") or []:
        uid, _ = resolve_unit(cv_index, u)
        if uid:
            used_ids.add(uid)
    for col in result.get("suggested_columns") or []:
        cid, _ = resolve_column(cv_index, col)
        if cid:
            used_ids.add(cid)
    concept = result.get("suggested_value_concept")
    if concept:
        cid, _ = resolve_value_concept(cv_index, concept)
        if cid:
            used_ids.add(cid)

    vendor_prefixes = [
        "ProteomeDiscoverer:", "MaxQuant:", "Mascot:", "SEQUEST:",
        "X!Tandem:", "Comet:", "MSFragger:", "Byonic:",
    ]

    cleaned = []
    seen_ids = set()
    for rel in rels:
        rid = rel.get("id", "")
        if not rid or rid in seen_ids:
            continue
        seen_ids.add(rid)

        if rid in used_ids:
            continue

        #fall back to the full cv
        term = id_to_term.get(rid) or cv_index["id_to_term"].get(rid, {})
        if term.get("is_obsolete"):
            continue

        term_name = term.get("name") or rel.get("name", "")
        if any(term_name.startswith(vp) for vp in vendor_prefixes):
            continue

        if overlap_levels.get(rid) == "low":
            continue

        cleaned.append(rel)

    cleaned.sort(key=lambda r: r.get("id", ""))
    result["suggested_relations"] = cleaned
    return result


def _canonicalize_returned_names(result, candidate_display_terms, cv_index):
    id_to_name = dict(cv_index["id_to_name"])
    id_to_name.update({t["id"]: t["name"] for t in candidate_display_terms})
    for item in result.get("overlap_results", []):
        canonical = id_to_name.get(item.get("id"))
        if canonical:
            item["name"] = canonical
    for rel in result.get("suggested_relations", []):
        canonical = id_to_name.get(rel.get("id"))
        if canonical:
            rel["name"] = canonical
    for col in result.get("suggested_columns") or []:
        if not isinstance(col, dict):
            continue
        canonical = (
            cv_index["column_id_to_name"].get(col.get("id"))
            or id_to_name.get(col.get("id"))
        )
        if canonical:
            col["name"] = canonical
    return result


def postprocess_result(result, candidate_raw, candidate_display, cv_index):
    if not isinstance(result, dict):
        return {}
    valid_ids = {t["id"] for t in candidate_raw}
    if valid_ids:
        result = _validate_returned_ids(result, valid_ids, cv_index)
    if candidate_display:
        result = _enforce_base_term_overlap_cap(result, candidate_display)
    result = _fix_result_name_casing(result)
    result = _enforce_value_type_rules(result, cv_index)
    if candidate_display:
        result = _stabilize_relations(result, candidate_display, cv_index)
    result = _canonicalize_returned_names(result, candidate_display, cv_index)
    return result


#streamlit UI

OVERLAP_DISPLAY_LIMIT = 10


def render_overlap_label(level):
    st.write(f"Overlap: **{level.upper()}**")


def render_classification_card(result):
    st.subheader("7-Dimension Classification")
    for dim_key, dim_info in CLASSIFICATION_SCHEMA.items():
        val = result.get(dim_key, "") or "not determined"
        st.write(f"**{dim_info['label']}:** {val}")


def render_compliance_card(checks):
    st.subheader("mzQC 1.0.0 specification compliance")
    st.caption(
        "These checks apply to the drafted term above, they say nothing about "
        "whether the term should exist -- read the verdict for that."
    )
    errors = [c for c in checks if c["level"] == "error"]
    warns = [c for c in checks if c["level"] == "warning"]
    if errors:
        st.error(
            f"{len(errors)} MUST-level problem(s) -- the term is not yet "
            "valid for an mzQC file."
        )
    elif warns:
        st.warning(
            f"No MUST-level problems, {len(warns)} SHOULD/RECOMMENDED "
            "item(s) to look at."
        )
    else:
        st.success("All checked MUST and SHOULD rules are satisfied.")
    for c in checks:
        marker = {"error": "MUST", "warning": "SHOULD", "ok": "OK"}[c["level"]]
        st.write(f"[{marker}] **{c['rule']}** -- {c['message']}")


def render_sidebar(display_terms, active_count, api_key, cv_meta, cv_index):
    st.sidebar.header("Configuration")
    if api_key:
        st.sidebar.success("API key loaded.")
    else:
        st.sidebar.error(
            "No API key found. Set OPENAI_API_KEY env var or "
            ".streamlit/secrets.toml."
        )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Controlled vocabulary")
    st.sidebar.write(f"Source: {cv_meta.get('origin', 'unknown')}")
    st.sidebar.write(
        f"data-version: {cv_meta.get('data-version', 'unknown')}"
    )
    if cv_meta.get("commit_sha"):
        st.sidebar.write(f"Commit: {cv_meta['commit_sha'][:10]}")
    if cv_meta.get("retrieved_at"):
        st.sidebar.write(f"Retrieved: {cv_meta['retrieved_at']}")
    if cv_meta.get("error"):
        st.sidebar.warning(f"Download problem: {cv_meta['error']}")
    st.sidebar.write(f"Terms parsed: {len(display_terms)}")
    st.sidebar.write(f"Active terms: {active_count}")
    st.sidebar.write(
        f"Category terms: {len(cv_index['category_name_to_id'])}"
    )
    st.sidebar.write(f"Column terms: {len(cv_index['used_column_id_to_name'])}")
    st.sidebar.write("Classification: 7 dimensions")
    st.sidebar.write(f"Model: {LLM_MODEL}")
    if st.sidebar.button("Re-download psi-ms.obo"):
        st.session_state.cv_refresh_token += 1
        st.cache_data.clear()
        st.rerun()
    with st.sidebar.expander("mzQC controlledVocabulary entry"):
        st.caption(
            "Paste this into the controlledVocabularies array of your mzQC "
            "file (spec 6.17/6.18)."
        )
        st.code(generate_cv_reference_snippet(cv_meta), language="json")
    used = st.session_state.get("request_count", 0)
    st.sidebar.write(
        f"Analyses used in this session: {used} / {MAX_REQUESTS_PER_SESSION}"
    )


def render_accepted_sidebar():
    accepted = load_accepted_metrics()
    st.sidebar.markdown("---")
    st.sidebar.header(f"Accepted New Metrics ({len(accepted)})")
    if accepted:
        for i, m in enumerate(accepted, 1):
            with st.sidebar.expander(
                f"{i}. {m.get('suggested_name', 'unnamed')}"
            ):
                ad = m.get("analytical_dimension")
                if ad:
                    st.write(f"**Analytical dim:** {ad}")
                vt = m.get("metric_value_type", "N/A")
                st.write(f"**Value type:** {vt}")
                lvl = m.get("max_overlap_level") or "none"
                st.write(f"**Max overlap:** {lvl.upper()}")
                rels = m.get("suggested_relations") or []
                if rels:
                    rel_strs = [
                        f"{r.get('id')} ({r.get('name')})" for r in rels
                    ]
                    st.write(f"**Relations:** {', '.join(rel_strs)}")
                new_cols = m.get("suggested_new_column_terms") or []
                if new_cols:
                    st.write(
                        "**New column terms:** "
                        + ", ".join(
                            c.get("name", "?") for c in new_cols
                            if isinstance(c, dict)
                        )
                    )
                if m.get("cv_version"):
                    st.caption(f"CV version: {m['cv_version']}")
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
    with st.expander(
        f"Browse existing vocabulary ({len(display_terms)} terms)",
        expanded=False,
    ):
        query = st.text_input(
            "Filter by name or accession", key="vocab_filter"
        ).strip().lower()
        shown = 0
        for t in display_terms:
            if query and query not in t["name"].lower() \
                    and query not in t["id"].lower():
                continue
            if shown >= 300:
                st.caption("... more terms hidden, refine the filter.")
                break
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
            shown += 1


def render_results(result, raw_terms, cv_index):
    st.markdown("---")
    _, is_duplicate, is_high, is_new_metric = compute_verdict_flags(result)
    _render_verdict(result, raw_terms, is_duplicate, is_high, is_new_metric)
    st.write(result.get("verdict_summary", ""))

    save_status = st.session_state.get("save_status")
    if save_status == "saved":
        st.write(
            "This metric has been saved to **accepted_new_metrics.json**. "
            "See the sidebar for all accepted metrics."
        )
    elif save_status == "already_saved":
        st.write("This metric was already saved previously.")
    elif save_status == "write_failed":
        st.warning(
            "The metric could not be written to accepted_new_metrics.json "
            "(the deployment filesystem may be read-only). Use the OBO block "
            "below directly."
        )

    if result.get("suggested_consolidation"):
        st.markdown("---")
        st.subheader("Consolidation suggestion")
        st.caption(
            "Parameters that always come out of the same computation are "
            "usually better expressed as one table metric with one column per "
            "parameter than as several single-value metrics."
        )
        st.write(result["suggested_consolidation"])

    has_dims = any(result.get(k) for k in CLASSIFICATION_SCHEMA)
    if has_dims:
        st.markdown("---")
        render_classification_card(result)

    if result.get("suggested_name"):
        obo_block, parts = generate_obo_block(result, cv_index)
        column_blocks = generate_new_column_blocks(parts, cv_index)
        checks = validate_generated_term(
            parts, cv_index, st.session_state.get("proposed_name", "")
        )
        resolution_note = (
            "Every accession below was resolved against the downloaded "
            "psi-ms.obo, unresolvable ones were dropped and are listed in "
            "the compliance report."
        )
        mzqc_note = (
            "How the metric would appear in the qualityMetrics array "
            "(spec 6.16 and section 7), with a placeholder value."
        )
        st.markdown("---")
        if is_new_metric:
            st.subheader("Generated OBO Entry")
            st.caption(resolution_note)
            st.code(obo_block, language=None)
            _render_new_column_blocks(column_blocks)
            st.markdown("---")
            st.subheader("mzQC qualityMetric example")
            st.caption(mzqc_note)
            st.code(generate_mzqc_snippet(parts, cv_index), language="json")
        else:
            #the compliance report refers to this block so it stays visible even when the verdict advises against submitting the term
            with st.expander(
                "Draft OBO entry for the proposal "
                "(shown for reference, the verdict above advises against "
                "submitting it as a new term)",
                expanded=False,
            ):
                st.caption(resolution_note)
                st.code(obo_block, language=None)
                _render_new_column_blocks(column_blocks)
                st.caption(mzqc_note)
                st.code(
                    generate_mzqc_snippet(parts, cv_index), language="json"
                )
        st.markdown("---")
        render_compliance_card(checks)

    _render_overlap_table(result, is_new_metric)
    _render_relations(result)
    return is_new_metric


def _render_new_column_blocks(column_blocks):
    if not column_blocks:
        return
    st.markdown("---")
    st.subheader(f"Draft column terms ({len(column_blocks)})")
    st.caption(
        "These columns do not exist in the CV yet. Each one is drafted as its "
        "own term under QC non-metric term, which is where column terms "
        "belong. Submit them together with the metric above: once they are "
        "accepted and given real accessions, replace the placeholder ids in "
        "the has_column lines."
    )
    for placeholder, name, block in column_blocks:
        st.write(f"**{placeholder}** -- {name}")
        st.code(block, language=None)


def _render_relations(result):
    """show suggested_relations"""
    rels = result.get("suggested_relations", [])
    if not rels:
        return
    st.markdown("---")
    st.subheader(f"Related Terms ({len(rels)})")
    st.caption(
        "These are existing vocabulary terms that the proposed metric "
        "references or aggregates. They are linked via has_relation."
    )
    for r in rels:
        st.write(f"**{r.get('id', '?')}** - {r.get('name', '?')}")


def _render_verdict(result, raw_terms, is_duplicate, is_high, is_new_metric):
    if result.get("needs_more_detail"):
        st.write(
            "**VERDICT:** More detail is needed to fully evaluate this "
            "proposal."
        )
    elif is_duplicate:
        st.write(
            "**VERDICT:** DUPLICATE -- this metric already exists in the "
            "vocabulary."
        )
        for item in result.get("overlap_results", []):
            if item.get("overlap_level", "").lower() == "duplicate":
                match_id = item.get("id", "")
                st.info(
                    f"Use existing term **{match_id}** "
                    f"({item.get('name')}) instead."
                )
                match_term = find_raw_term_by_id(raw_terms, match_id)
                if match_term:
                    st.subheader("Existing Term OBO Entry")
                    st.code(reconstruct_obo_block(match_term), language=None)
                break
    elif is_high:
        st.write(
            "**VERDICT:** HIGH OVERLAP -- this may be redundant with an "
            "existing term."
        )
        for item in result.get("overlap_results", []):
            if item.get("overlap_level", "").lower() == "high":
                match_id = item.get("id", "")
                st.warning(
                    f"Closely related to **{match_id}** "
                    f"({item.get('name')}). "
                    "Consider whether a new term is needed."
                )
                match_term = find_raw_term_by_id(raw_terms, match_id)
                if match_term:
                    with st.expander(
                        f"View existing term {match_id}", expanded=False
                    ):
                        st.code(
                            reconstruct_obo_block(match_term), language=None
                        )
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
            st.write(
                f"**{item.get('id', '?')}** -- "
                f"{item.get('name', '?')}{obs_tag}"
            )
            render_overlap_label(lvl)
            st.caption(item.get("reasoning", ""))
            st.write("")
    elif is_new_metric and not result.get("needs_more_detail"):
        st.markdown("---")
        st.write(
            "This metric does not overlap with any existing term in the "
            "PSI-MS vocabulary."
        )


def render_clarification_ui(result, api_key, cv_index, cv_meta):
    if not (
        result.get("needs_more_detail")
        and result.get("clarification_question")
    ):
        return False
    st.markdown("---")
    st.subheader("Clarification Needed")
    st.write(result["clarification_question"])
    follow_up = st.text_area(
        "Provide additional detail:", key="followup_text", height=100
    )
    if st.button("Re-Analyze"):
        if not follow_up.strip():
            st.warning("Please type some additional detail first.")
            return False
        #reanalysis is a full LLM call, so it counts against the cap
        if not check_rate_limit():
            st.error(
                f"Session limit reached ({MAX_REQUESTS_PER_SESSION} "
                "analyses). Reload the page or try again later."
            )
            return False
        st.session_state.messages.append({
            "role": "user",
            "content": (
                f'Additional information:\n\n"{follow_up}"\n\n'
                "Please re-evaluate the proposed metric."
            ),
        })
        with st.spinner(f"Re-analyzing with {LLM_MODEL} ..."):
            result2, error2 = call_gpt(
                st.session_state.messages, api_key
            )
            if error2:
                st.error(error2)
                return False
            #the quota is only spent once we actually have an answer
            st.session_state.request_count += 1
            candidate_raw = st.session_state.get("candidate_terms_raw", [])
            candidate_display = [
                term_to_display_dict(t, cv_index) for t in candidate_raw
            ]
            result2 = postprocess_result(
                result2, candidate_raw, candidate_display, cv_index
            )
            st.session_state.result = result2
            st.session_state.save_status = persist_if_new(result2, cv_meta)
            st.session_state.messages.append({
                "role": "assistant",
                "content": json.dumps(result2),
            })
            st.rerun()
    return False


#data loading

@st.cache_resource(show_spinner=False)
def load_metrics(_cv_text, cache_key):
    """parse the downloaded CV once per CV version -> cache_resource not cache_data"""
    raw_terms = parse_obo_text(_cv_text)
    cv_index = build_cv_index(raw_terms)
    display_terms = [term_to_display_dict(t, cv_index) for t in raw_terms]
    return raw_terms, display_terms, cv_index


@st.cache_resource
def load_knowledge_graph(_raw_terms, cache_key):
    return build_knowledge_graph(_raw_terms)


@st.cache_resource
def load_embeddings(_raw_terms, api_key, cache_key):
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
    st.set_page_config(
        page_title="QC Metric Proposal Analyzer", layout="wide"
    )

    for key, default in [
        ("messages", []), ("result", None),
        ("proposed_name", ""), ("proposed_desc", ""),
        ("candidate_terms_raw", []),
        ("request_count", 0), ("first_request_time", time.time()),
        ("cv_refresh_token", 0), ("save_status", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    with st.spinner("Downloading the current psi-ms.obo ..."):
        cv_text, cv_meta = fetch_cv_text(st.session_state.cv_refresh_token)

    if not cv_text.strip():
        st.error(
            "Could not obtain psi-ms.obo. The app tried "
            f"{OBO_RAW_URL}, the local cache in {CV_CACHE_DIR}, and "
            f"{LOCAL_OBO_FALLBACK}. Check the network connection or place a "
            "copy of psi-ms.obo next to this script."
        )
        return

    raw_terms, display_terms, cv_index = load_metrics(
        cv_text, _text_hash(cv_text)
    )
    if not raw_terms:
        st.error("The downloaded psi-ms.obo contained no parsable terms.")
        return

    #cache key for the resources that must be rebuilt when the cv changes
    cv_cache_key = _obo_content_hash(raw_terms)

    active_count = sum(1 for t in display_terms if not t.get("is_obsolete"))
    qc_count = sum(1 for t in display_terms if _is_in_qc_namespace(t["id"]))
    api_key = _resolve_api_key()

    st.title("PSI-MS QC Metric Proposal Analyzer")
    st.write(
        f"Validates new metric proposals against **{len(display_terms)}** "
        f"terms ({active_count} active, {len(display_terms) - active_count} "
        f"obsolete, {qc_count} in the QC namespace) parsed from "
        f"**psi-ms.obo data-version "
        f"{cv_meta.get('data-version', 'unknown')}**, downloaded from the "
        f"HUPO-PSI psi-ms-CV repository, using **{LLM_MODEL}** and the "
        f"mzQC 1.0.0 specification rules."
    )

    render_sidebar(display_terms, active_count, api_key, cv_meta, cv_index)
    render_accepted_sidebar()

    G, term_ids, embedding_matrix = None, None, None
    if api_key:
        G = load_knowledge_graph(raw_terms, cv_cache_key)
        with st.spinner("Preparing term embeddings ..."):
            term_ids, embedding_matrix = load_embeddings(
                raw_terms, api_key, cv_cache_key
            )

    st.subheader("Propose a New QC Metric")
    proposed_name = st.text_input(
        "Proposed metric name",
        placeholder="e.g. MS2 spectral entropy median",
    )
    proposed_desc = st.text_area(
        "Description -- explain what this metric measures and how it is "
        "computed",
        placeholder=(
            "Include:\n"
            "- What quantity is being measured\n"
            "- How it is computed (mean, median, ratio, count, etc.)\n"
            "- Whether it requires identification results (ID-based) or "
            "not (ID-free)\n"
            "- What MS level (MS1, MS2, run-level)\n"
            "- Whether it summarises one run or several runs\n"
            "- What unit (ppm, seconds, fraction, count, etc.)\n"
            "- What value type (single value, n-tuple, table, matrix), and "
            "for a table, which columns"
        ),
        height=180,
    )

    can_submit = bool(proposed_name and proposed_desc and api_key)
    analyze_clicked = st.button("Analyze Proposal", disabled=not can_submit)

    if not api_key and (proposed_name or proposed_desc):
        st.warning(
            "No API key configured. Set OPENAI_API_KEY env var or "
            ".streamlit/secrets.toml."
        )

    if analyze_clicked:
        if not check_rate_limit():
            st.error(
                f"Session limit reached ({MAX_REQUESTS_PER_SESSION} "
                "analyses). Reload the page or try again later."
            )
            return

        st.session_state.proposed_name = proposed_name
        st.session_state.proposed_desc = proposed_desc
        st.session_state.save_status = None

        with st.spinner("Retrieving relevant terms ..."):
            candidate_raw = retrieve_relevant_terms(
                proposed_name, proposed_desc, raw_terms, G,
                term_ids, embedding_matrix, api_key,
                embed_quota=EMBED_QUOTA, graph_quota=GRAPH_QUOTA,
                hops=GRAPH_HOP_LIMIT,
            )
            candidate_display = [
                term_to_display_dict(t, cv_index) for t in candidate_raw
            ]
            st.session_state.candidate_terms_raw = candidate_raw

        system_prompt = build_system_prompt(
            candidate_display, len(display_terms), active_count,
            cv_index, cv_meta,
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

        with st.spinner(f"Analyzing with {LLM_MODEL} ..."):
            result, error = call_gpt(
                st.session_state.messages, api_key
            )
            if error:
                st.error(error)
                return

            st.session_state.request_count += 1

            result = postprocess_result(
                result, candidate_raw, candidate_display, cv_index
            )

            st.session_state.result = result

            st.session_state.save_status = persist_if_new(result, cv_meta)
            st.session_state.messages.append({
                "role": "assistant",
                "content": json.dumps(result),
            })

    result = st.session_state.result
    if result is None:
        render_vocabulary_browser(display_terms)
        return

    render_results(result, raw_terms, cv_index)
    render_clarification_ui(result, api_key, cv_index, cv_meta)


if __name__ == "__main__":
    main()
