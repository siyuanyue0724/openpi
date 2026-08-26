"""Training-data boundary adapters for PICF-Next.

Submodules are deliberately not imported here. Raster target generation only
needs NumPy, while learned object-target construction needs PyTorch; importing
the package must not force the heavier runtime into offline data jobs.
"""
