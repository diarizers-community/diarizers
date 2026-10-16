"""Package setup for diarizers v1.0.0."""

from setuptools import find_packages
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
  LONG_DESCRIPTION = fh.read()

INSTALL_REQUIRES = [
    "numpy>=1.22.0",
    "scipy>=1.8.0",
    "soundfile>=0.12.0",
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "huggingface-hub>=0.20.0",
]

EXTRAS_REQUIRE = {
    "llm": [
        "llama-cpp-python>=0.2.80",
        "peft>=0.10.0",
        "trl>=0.8.0",
        "datasets>=2.16.0",
    ],
    "audio": [
        "torchaudio>=2.0.0",
        "spectralcluster>=0.2.16",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "coverage>=7.0.0",
        "flake8>=6.0.0",
        "build>=1.0.0",
        "twine>=5.0.0",
    ],
}

setup(
    name="diarizers",
    version="1.0.0",
    description=(
        "Unified Open-Source ASR, Speaker Diarization, and DiarizationLM "
        "Post-Processing Pipeline"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords=(
        "speech recognition, speaker diarization, diarizationlm, llm, "
        "whisper, ecapa-tdnn, spectral clustering, gemma"
    ),
    license="Apache-2.0",
    author="Kamil Akesbi, Quan Wang, Sanchit Gandhi",
    url="https://github.com/diarizers-community/diarizers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.9.0",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "diarizers=diarizers.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
