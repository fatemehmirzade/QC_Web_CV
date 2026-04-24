import os
import json
import re
import datetime
import streamlit as st
from openai import OpenAI

NEW_METRICS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "accepted_new_metrics.json"
)

OVERLAP_RANK = {
    "duplicate": 4,
    "high": 3,
    "moderate": 2,
    "low": 1,
}


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


def max_overlap_level(overlap_results):
    best = None
    best_rank = 0
    for r in overlap_results:
        lvl = r.get("overlap_level", "low").lower()
        rank = OVERLAP_RANK.get(lvl, 0)
        if rank > best_rank:
            best_rank = rank
            best = lvl
    return best


def parse_obo(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    raw_blocks = re.split(r"\n(?=\[Term\])", text)
    terms = []

    for block in raw_blocks:
        if not block.strip().startswith("[Term]"):
            continue

        term = {
            "id": None,
            "name": None,
            "def": None,
            "def_xrefs": [],
            "comment": None,
            "synonyms": [],
            "is_a": [],
            "is_obsolete": False,
            "replaced_by": None,
            "categories": [],
            "units": [],
            "xsd_value_type": None,
            "value_concepts": [],
            "columns": [],
            "optional_columns": [],
            "relations": [],
            "order": None,
            "domain": None,
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
                    term["def_xrefs"] = [
                        r.strip()
                        for r in m.group(2).split(",")
                        if r.strip()
                    ]
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
                rel = line[14:].strip()
                if rel.startswith("has_metric_category "):
                    term["categories"].append(rel[20:].strip())
                elif rel.startswith("has_units "):
                    term["units"].append(rel[10:].strip())
                elif rel.startswith("has_value_type "):
                    term["xsd_value_type"] = rel[15:].strip()
                elif rel.startswith("has_value_concept "):
                    term["value_concepts"].append(rel[18:].strip())
                elif rel.startswith("has_column "):
                    term["columns"].append(rel[11:].strip())
                elif rel.startswith("has_optional_column "):
                    term["optional_columns"].append(rel[20:].strip())
                elif rel.startswith("has_relation "):
                    term["relations"].append(rel[13:].strip())
                elif rel.startswith("has_order "):
                    term["order"] = rel[10:].strip()
                elif rel.startswith("has_domain "):
                    term["domain"] = rel[11:].strip()

        if term["id"] and term["name"]:
            terms.append(term)

    return terms


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
    parts = ref_str.split(" ! ", 1)
    return parts[0].strip()


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


@st.cache_data
def load_metrics():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    obo_path = os.path.join(script_dir, "qc_metrics_specific.obo")
    if not os.path.exists(obo_path):
        st.error(
            f"OBO file not found at {obo_path}. "
            "Place qc_metrics_specific.obo next to this script."
        )
        return [], []
    raw_terms = parse_obo(obo_path)
    display_terms = [term_to_display_dict(t) for t in raw_terms]
    return raw_terms, display_terms


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
    record["max_overlap_level"] = max_overlap_level(
        result.get("overlap_results", [])
    )
    record["overlap_results"] = result.get("overlap_results", [])
    record["verdict_summary"] = result.get("verdict_summary")
    record["saved_at"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()

    accepted.append(record)
    save_accepted_metrics(accepted)
    return True, "saved"


def build_classification_prompt_section():
    lines = []
    lines.append(
        "Each QC metric must be classified along seven independent "
        "dimensions. For each dimension, choose exactly one value."
    )
    lines.append("")

    for i, (dim_key, dim_info) in enumerate(
        CLASSIFICATION_SCHEMA.items(), 1
    ):
        lines.append(
            f"  Dimension {i} -- {dim_info['label']} "
            f"(JSON key: \"{dim_key}\"):"
        )
        lines.append(f"  {dim_info['description']}")
        for val_name, val_desc in dim_info["values"].items():
            lines.append(f"    - \"{val_name}\": {val_desc}")
        lines.append("")

    lines.append("CRITICAL DISAMBIGUATION RULES:")
    lines.append("")
    lines.append(
        "  workflow_stage vs data_dependency -- these are INDEPENDENT:"
    )
    lines.append(
        "    workflow_stage = where the MEASURED QUANTITY physically "
        "originates in the pipeline."
    )
    lines.append(
        "    data_dependency = what INPUT DATA is needed to COMPUTE "
        "the metric."
    )
    lines.append(
        "    Example: 'precursor ppm deviation mean' measures MS1 "
        "mass accuracy (workflow_stage = MS1 acquisition stage) but "
        "requires identification results as a filter "
        "(data_dependency = identification results)."
    )
    lines.append(
        "    Use 'identification stage' ONLY when the quantity itself "
        "is an identification quality measure (FDR, ID rate, PSM count)."
    )
    lines.append(
        "    Use 'quantification stage' ONLY when the quantity itself "
        "is a quantitative accuracy/precision measure."
    )
    lines.append("")
    lines.append("  quality_directionality -- decision rules:")
    lines.append(
        "    1. Purity, coverage, rate, completeness fraction where "
        "1.0 = perfect -> \"higher is better\"."
    )
    lines.append(
        "    2. Error, deviation, contamination, bad event count "
        "-> \"lower is better\"."
    )
    lines.append(
        "    3. SIGNED deviation centered on a target (mass deviation "
        "where 0 ppm is ideal, temperature) -> \"target range\"."
    )
    lines.append(
        "    4. \"context dependent\" ONLY when direction genuinely "
        "varies by experimental design."
    )
    lines.append("")

    return "\n".join(lines)


def build_system_prompt(display_terms):
    metrics_text = json.dumps(display_terms, indent=1)
    active_count = sum(
        1 for t in display_terms if not t.get("is_obsolete")
    )
    total_count = len(display_terms)

    classification_section = build_classification_prompt_section()
    legacy_cats = ", ".join(
        f'"{c}"' for c in CATEGORY_NAME_TO_ID.keys()
    )
    dim_keys_example = ",\n".join(
        f'  "{k}": "..."' for k in CLASSIFICATION_SCHEMA
    )

    return f"""You are an expert in mass spectrometry quality control metrics, specifically the PSI-MS controlled vocabulary used in mzQC files.

Below is the COMPLETE list of all existing QC terms parsed from the official OBO ontology ({total_count} terms total, {active_count} active, {total_count - active_count} obsolete). Each entry has:
  - id, name, def, comment, synonyms
  - value_type: "single value", "n-tuple"/"tuple", "table", "matrix"
  - categories: legacy OBO metric categories
  - units: measurement units
  - value_concepts: statistical concepts (median, standard deviation, quantile, etc.)
  - relations: list of related CV terms via has_relation (e.g. linking a metric to the base quantity it summarizes, or to other metrics that share the same underlying data)
  - columns / optional_columns: for table-type metrics, what columns are required/optional
  - order: ordering hint (e.g. "lower score better")
  - domain: value domain constraint (e.g. "value between 0 and 1 inclusive")
  - is_obsolete, replaced_by

<existing_qc_metrics>
{metrics_text}
</existing_qc_metrics>

A user is proposing a new QC metric. Perform the following steps.

STEP 1 -- ASSESS DESCRIPTION QUALITY:
It must specify: (a) what quantity is measured, (b) how it is computed, (c) what MS level, (d) ID-based or ID-free, (e) unit/value type.
If any are missing, set needs_more_detail to true and write a specific clarification_question. Still provide best attempt at all other fields.

STEP 2 -- GENERATE FORMAL NAME:
Short, precise, lowercase (except proper nouns). Follow patterns: "X distribution mean", "number of X", "X quantiles", etc. No MS:4000XXX id.

STEP 3 -- GENERATE FORMAL DEFINITION:
OBO-style def: "..." text. Precise, technical, self-contained. Reference other MS terms by accession where relevant.

STEP 4 -- SEVEN-DIMENSION CLASSIFICATION:
{classification_section}

STEP 5 -- LEGACY OBO FIELDS:
  - suggested_categories: list from: {legacy_cats}.
  - suggested_units: list of unit strings.
  - suggested_xsd_type: "xsd:float" or "xsd:int".

STEP 6 -- RELATIONSHIPS (has_relation):
Identify any existing CV terms that are semantically related to the proposed metric via has_relation. These are terms that:
  - Define the base quantity being summarized (e.g. MS:1000285 "total ion current" for a TIC-based metric).
  - Define a parent concept the metric is derived from (e.g. MS:1000505 "base peak intensity" for a BPI metric).
  - Are companion metrics computed from the same data (e.g. linking a charge fraction table to its corresponding ratio metrics).
  - Are referenced in the definition by accession.

Report as suggested_relations: a list of objects with "id" and "name".
Only include terms where there is a genuine semantic link, not just topical similarity. If no has_relation is appropriate, use an empty list.

STEP 7 -- TABLE COLUMNS (if metric_value_type is "table"):
If the metric is a table, specify:
  - suggested_columns: list of column descriptors (e.g. "charge state", "fraction").
Otherwise omit or use an empty list.

STEP 8 -- OVERLAP ANALYSIS (BE THOROUGH AND CONSISTENT):
Compare against every existing term by semantic meaning. Use all fields including relations.

OVERLAP LEVEL DEFINITIONS:
  "duplicate"  = SAME quantity, SAME statistic, SAME scope, SAME filtering, SAME value type.
  "high"       = same quantity but output is derivable from or contained in an existing term.
  "moderate"   = related quantity at the same MS level (mean vs sigma, precursor vs fragment).
  "low"        = loosely related concept.

CONSISTENCY RULES:
  - mean vs sigma of same distribution = "moderate" (different properties).
  - single-value statistic already contained in an existing distribution/quantile term = "high".
  - Sharing a unit or category alone is NOT enough for overlap.

For each overlapping term report: id, name, overlap_level, is_obsolete, reasoning.

STEP 9 -- VERDICT:
  - Any "duplicate": metric IS a duplicate.
  - Highest "high": flag HIGH OVERLAP.
  - Otherwise: metric is NEW.

RESPONSE FORMAT -- ONLY valid JSON, no markdown fences:

{{
  "needs_more_detail": false,
  "clarification_question": null,
  "suggested_name": "metric name",
  "suggested_def": "OBO definition text.",

{dim_keys_example},

  "suggested_categories": ["ID free metric", "MS2 metric"],
  "suggested_units": ["parts per million"],
  "suggested_xsd_type": "xsd:float",
  "suggested_relations": [
    {{"id": "MS:4000072", "name": "observed mass accuracy"}}
  ],
  "suggested_columns": [],

  "overlap_results": [
    {{
      "id": "MS:4000026",
      "name": "fragment ppm deviation median",
      "overlap_level": "moderate",
      "is_obsolete": false,
      "reasoning": "Both measure mass accuracy deviation but at different MS levels."
    }}
  ],
  "is_new": true,
  "verdict_summary": "This metric is new."
}}

Use the EXACT string values for each dimension. Do not invent new values."""


def call_gpt(messages, api_key, reasoning_effort):
    client = OpenAI(api_key=api_key)
    kwargs = dict(
        model="gpt-5.2",
        messages=messages,
        max_completion_tokens=16000,
    )

    response = None
    if reasoning_effort in ("low", "medium", "high"):
        try:
            response = client.chat.completions.create(
                **kwargs, reasoning_effort=reasoning_effort
            )
        except (TypeError, Exception) as e:
            if "reasoning" in str(e).lower() or isinstance(e, TypeError):
                response = None
            else:
                raise

    if response is None:
        response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content.strip()

    cleaned = raw
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass

    depth = 0
    start = None
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
        obo_lines.append(
            f"relationship: has_value_type {xsd_type} ! "
            f"The allowed value-type for this CV term"
        )

    for unit in units:
        uid = UNIT_NAME_TO_ID.get(unit)
        if uid:
            obo_lines.append(
                f"relationship: has_units {uid} ! {unit}"
            )
        else:
            obo_lines.append(
                f"relationship: has_units ... ! {unit}"
            )

    for cat_name in categories:
        cat_id = CATEGORY_NAME_TO_ID.get(cat_name)
        if cat_id:
            obo_lines.append(
                f"relationship: has_metric_category "
                f"{cat_id} ! {cat_name}"
            )

    for rel in relations:
        rel_id = rel.get("id", "")
        rel_name = rel.get("name", "")
        if rel_id:
            obo_lines.append(
                f"relationship: has_relation "
                f"{rel_id} ! {rel_name}"
            )

    for col in columns:
        obo_lines.append(
            f"relationship: has_column ... ! {col}"
        )

    return "\n".join(obo_lines)


def render_overlap_label(level):
    st.write(f"Overlap: **{level.upper()}**")


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
        lines.append(
            f"relationship: has_value_type "
            f"{raw_term['xsd_value_type']}"
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
    if raw_term.get("order"):
        lines.append(f"relationship: has_order {raw_term['order']}")
    if raw_term.get("domain"):
        lines.append(f"relationship: has_domain {raw_term['domain']}")
    if raw_term.get("is_obsolete"):
        lines.append("is_obsolete: true")
    if raw_term.get("replaced_by"):
        lines.append(f"replaced_by: {raw_term['replaced_by']}")
    return "\n".join(lines)


def find_raw_term_by_id(raw_terms, term_id):

    for t in raw_terms:
        if t["id"] == term_id:
            return t
    return None


def render_classification_card(result):
    st.subheader("7-Dimension Classification")
    for dim_key, dim_info in CLASSIFICATION_SCHEMA.items():
        val = result.get(dim_key, "")
        if not val:
            val = "not determined"
        st.write(f"**{dim_info['label']}:** {val}")


def main():
    st.set_page_config(
        page_title="QC Metric Proposal Analyzer", layout="wide"
    )

    raw_terms, display_terms = load_metrics()
    if not raw_terms:
        return

    active_count = sum(
        1 for t in display_terms if not t.get("is_obsolete")
    )
    obsolete_count = len(display_terms) - active_count

    st.title("PSI-MS QC Metric Proposal Analyzer")
    st.write(
        f"Validates new metric proposals against "
        f"**{len(display_terms)}** existing QC terms "
        f"({active_count} active, {obsolete_count} obsolete) parsed "
        f"from the official OBO ontology using **GPT-5.2** with "
        f"configurable reasoning."
    )

    st.sidebar.header("Configuration")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            api_key = ""

    if api_key:
        st.sidebar.success("API key loaded.")
    else:
        st.sidebar.error(
            "No API key found. Set the OPENAI_API_KEY environment "
            "variable or add it to .streamlit/secrets.toml."
        )

    reasoning_effort = st.sidebar.radio(
        "Reasoning Effort",
        options=["low", "medium", "high"],
        index=1,
        help=(
            "low = faster, less thorough.  |  "
            "medium = balanced.  |  "
            "high = slowest, most accurate."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Source: qc_metrics_specific.obo")
    st.sidebar.write(
        f"Terms: {len(display_terms)} ({active_count} active)"
    )
    st.sidebar.write("Classification: 7 dimensions")
    st.sidebar.write("Model: GPT-5.2")
    st.sidebar.write(f"Reasoning: {reasoning_effort}")

    st.sidebar.markdown("---")
    accepted = load_accepted_metrics()
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
                rels = m.get("suggested_relations", [])
                if rels:
                    rel_strs = [
                        f"{r['id']} ({r['name']})" for r in rels
                    ]
                    st.write(
                        f"**Relations:** {', '.join(rel_strs)}"
                    )
                st.caption(
                    f"Saved: {m.get('saved_at', 'unknown')}"
                )
                st.code(m.get("suggested_def", ""), language=None)

        st.sidebar.download_button(
            label="Download accepted_new_metrics.json",
            data=json.dumps(accepted, indent=2),
            file_name="accepted_new_metrics.json",
            mime="application/json",
        )
    else:
        st.sidebar.caption("No new metrics accepted yet.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "result" not in st.session_state:
        st.session_state.result = None
    if "proposed_name" not in st.session_state:
        st.session_state.proposed_name = ""
    if "proposed_desc" not in st.session_state:
        st.session_state.proposed_desc = ""

    system_prompt = build_system_prompt(display_terms)

    st.subheader("Propose a New QC Metric")

    proposed_name = st.text_input(
        "Proposed metric name",
        placeholder="e.g. MS2 spectral entropy median",
    )

    proposed_desc = st.text_area(
        "Description -- explain what this metric measures and "
        "how it is computed",
        placeholder=(
            "Include:\n"
            "- What quantity is being measured\n"
            "- How it is computed (mean, median, ratio, count, etc.)\n"
            "- Whether it requires identification results "
            "(ID-based) or not (ID-free)\n"
            "- What MS level (MS1, MS2, run-level)\n"
            "- What unit (ppm, seconds, fraction, count, etc.)"
        ),
        height=160,
    )

    can_submit = bool(proposed_name and proposed_desc and api_key)
    analyze_clicked = st.button(
        "Analyze Proposal", disabled=not can_submit
    )

    if not api_key and (proposed_name or proposed_desc):
        st.warning(
            "No API key configured. Set OPENAI_API_KEY as an "
            "environment variable or in .streamlit/secrets.toml."
        )

    if analyze_clicked:
        st.session_state.proposed_name = proposed_name
        st.session_state.proposed_desc = proposed_desc
        st.session_state.messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"I want to propose a new QC metric for the "
                    f"PSI-MS controlled vocabulary.\n\n"
                    f"Proposed name: \"{proposed_name}\"\n\n"
                    f"Description: \"{proposed_desc}\""
                ),
            },
        ]

        with st.spinner(
            f"Analyzing with GPT-5.2 "
            f"(reasoning: {reasoning_effort}) ..."
        ):
            result, error = call_gpt(
                st.session_state.messages,
                api_key,
                reasoning_effort,
            )
            if error:
                st.error(error)
                return
            st.session_state.result = result
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(result),
                }
            )

    result = st.session_state.result
    if result is None:
        with st.expander(
            f"Browse existing QC vocabulary "
            f"({len(display_terms)} terms)",
            expanded=False,
        ):
            for t in display_terms:
                cats = t.get("categories", [])
                obs = " [OBSOLETE]" if t.get("is_obsolete") else ""
                cat_str = ", ".join(cats) if cats else ""
                rels = t.get("relations", [])
                rel_str = ""
                if rels:
                    rel_str = (
                        "  ->  "
                        + ", ".join(
                            f"{r['id']} ({r['name']})"
                            for r in rels
                        )
                    )
                label = f"{t['id']}  |  {t['name']}{obs}"
                if cat_str:
                    label += f"  |  [{cat_str}]"
                label += rel_str
                st.text(label)
        return

    st.markdown("---")

    top_level = max_overlap_level(
        result.get("overlap_results", [])
    )
    is_duplicate = top_level == "duplicate"
    is_high = top_level == "high"
    is_new_metric = (
        not result.get("needs_more_detail")
        and result.get("is_new")
        and top_level not in ("duplicate", "high")
    )

    if result.get("needs_more_detail"):
        st.write(
            "**VERDICT:** More detail is needed to fully "
            "evaluate this proposal."
        )
    elif is_duplicate:
        st.write(
            "**VERDICT:** DUPLICATE -- this metric already "
            "exists in the vocabulary."
        )
        for item in result.get("overlap_results", []):
            if item.get("overlap_level", "").lower() == "duplicate":
                match_id = item.get("id", "")
                st.info(
                    f"Use existing term **{match_id}** "
                    f"({item.get('name')}) instead."
                )
                match_term = find_raw_term_by_id(
                    raw_terms, match_id
                )
                if match_term:
                    st.subheader("Existing Term OBO Entry")
                    st.code(
                        reconstruct_obo_block(match_term),
                        language=None,
                    )
                break
    elif is_high:
        st.write(
            "**VERDICT:** HIGH OVERLAP -- this may be redundant "
            "with an existing term."
        )
        for item in result.get("overlap_results", []):
            if item.get("overlap_level", "").lower() == "high":
                match_id = item.get("id", "")
                st.warning(
                    f"Closely related to **{match_id}** "
                    f"({item.get('name')}). Consider whether a "
                    f"new term is needed."
                )
                match_term = find_raw_term_by_id(
                    raw_terms, match_id
                )
                if match_term:
                    with st.expander(
                        f"View existing term {match_id}",
                        expanded=False,
                    ):
                        st.code(
                            reconstruct_obo_block(match_term),
                            language=None,
                        )
                break
    elif is_new_metric:
        st.write("**VERDICT:** NEW METRIC -- no duplicate found.")
    else:
        st.write("**VERDICT:** Likely a new metric.")

    st.write(result.get("verdict_summary", ""))

    if is_new_metric and result.get("suggested_name"):
        saved, status = append_new_metric(
            result,
            st.session_state.get("proposed_name", ""),
            st.session_state.get("proposed_desc", ""),
        )
        if saved:
            st.write(
                "This metric has been saved to "
                "**accepted_new_metrics.json**. "
                "See the sidebar for all accepted metrics."
            )
        elif status == "already_saved":
            st.write("This metric was already saved previously.")

    if (
        result.get("needs_more_detail")
        and result.get("clarification_question")
    ):
        st.markdown("---")
        st.subheader("Clarification Needed")
        st.write(result["clarification_question"])

        follow_up = st.text_area(
            "Provide additional detail:",
            key="followup_text",
            height=100,
        )
        if st.button("Re-Analyze"):
            if not follow_up.strip():
                st.warning(
                    "Please type some additional detail first."
                )
            else:
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Additional information:\n\n"
                            f"\"{follow_up}\"\n\n"
                            f"Please re-evaluate the proposed metric."
                        ),
                    }
                )
                with st.spinner(
                    f"Re-analyzing with GPT-5.2 "
                    f"(reasoning: {reasoning_effort}) ..."
                ):
                    result2, error2 = call_gpt(
                        st.session_state.messages,
                        api_key,
                        reasoning_effort,
                    )
                    if error2:
                        st.error(error2)
                    else:
                        st.session_state.result = result2
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps(result2),
                            }
                        )
                        st.rerun()

    has_dims = any(result.get(k) for k in CLASSIFICATION_SCHEMA)
    if has_dims:
        st.markdown("---")
        render_classification_card(result)

    if is_new_metric and result.get("suggested_name"):
        st.markdown("---")
        st.subheader("Generated OBO Entry")

        obo_block = generate_obo_block(result)
        st.code(obo_block, language=None)

    overlaps = result.get("overlap_results", [])
    if overlaps:
        st.markdown("---")
        st.subheader(
            f"Overlap Analysis -- {len(overlaps)} related term(s)"
        )

        level_order = ["duplicate", "high", "moderate", "low"]
        sorted_overlaps = sorted(
            overlaps,
            key=lambda x: (
                level_order.index(
                    x.get("overlap_level", "low").lower()
                )
                if x.get("overlap_level", "low").lower()
                in level_order
                else 99
            ),
        )

        for item in sorted_overlaps:
            lvl = item.get("overlap_level", "low").lower()
            obs_tag = (
                " (OBSOLETE)" if item.get("is_obsolete") else ""
            )
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
            "This metric does not overlap with any existing "
            "term in the PSI-MS vocabulary."
        )


if __name__ == "__main__":
    main()
