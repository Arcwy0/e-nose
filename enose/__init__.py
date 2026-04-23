"""E-Nose multimodal vision-smell system.

Package layout:
    config        - central constants, sensor lists, paths
    sensors       - hardware I/O (e-nose, UART env) + transforms
    classifier    - smell classifiers (Balanced RF, SGD, XGBoost) + preprocessing
    vision        - Florence-2 VLM loading + inference
    visualization - plots: confusion matrix, features, env sensors, data quality
    server        - FastAPI app, schemas, routes
    client        - interactive client: sensors + camera + REST + pipeline
    utils         - CSV I/O, logging helpers
"""

__version__ = "3.1.0"
