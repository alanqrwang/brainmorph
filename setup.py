import setuptools
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, "brainmorph"))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brainmorph",
    version="1.0.0",
    author="Alan Q. Wang",
    author_email="alanqrwang@gmail.com",
    url="https://github.com/alanqrwang/brainmorph",
    description="BrainMorph is a foundational keypoint model for robust and flexible brain MRI registration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "keymorph>=2.0.0",
        "numpy>=1.19.1",
        "ogb>=1.2.6",
        "outdated>=0.2.0",
        "pandas>=1.1.0",
        "ogb>=1.2.6",
        "pytz>=2020.4",
        "torch>=1.7.0",
        "torchvision>=0.8.2",
        "scikit-learn>=0.20.0",
        "scipy>=1.5.4",
        "torchio>=0.19.6",
        "gdown>=5.2.0",
        "scikit-image>=0.22.0",
        "matplotlib>=3.8.3",
        "h5py>=3.10.0",
    ],
    license="MIT",
    packages=setuptools.find_packages(
        exclude=[
            "baselines",
            "scripts",
            "dataset",
            "tests",
        ]
    ),
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)
