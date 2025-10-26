from setuptools import setup, find_packages

setup(
    name='scientific-image-analyzer',
    version='1.0.0',
    author='Scientific Computing Team',
    description='Multi-dimensional microscopy image analysis suite',
    packages=find_packages(),
    py_modules=[
        "advanced_analysis",
        "batch_processor",
        "fibsem_plugins",
        "file_io_napari",
        "laminAmutants",
        "main",
        "main_napari",
        "scientific_analyzer",
        "simple_analyzer",
        "tif_diagnostic",
        "timelapse_processor",
    ],
    # By using include_package_data=True and a MANIFEST.in file,
    # we ensure that non-Python files like napari.yaml are included.
    include_package_data=True,
    install_requires=[
        'napari',
        'aicsimageio',
        'dask',
        'tifffile',
        'scikit-image',
        'numpy',
        'Pillow',
    ],
    entry_points={
        'napari.manifest': [
            'scientific-image-analyzer = napari.yaml',
        ],
    },
)
