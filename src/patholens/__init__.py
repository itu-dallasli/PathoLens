"""
PathoLens — A Traceable Retrieval-Based Clinical Decision Support System
for Breast Cancer Histopathology.

Modules:
    preprocessing   – WSI ingestion, tissue segmentation, patch extraction
    embedding       – UNI feature extractor
    sequence_model  – Mamba/SAMBA slide encoder
    retrieval       – CMEA hierarchical case retrieval
    entity_extraction – KARG clinical entity extraction
    explainability  – RAAF entity-guided explainability
    report_generation – CSAL FHIR report builder
    pipeline        – End-to-end inference orchestration
    api             – FastAPI web service
"""

__version__ = "0.1.0"
