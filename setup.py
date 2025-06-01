"""
Setup script for StackDB - In-Memory Vector Database
"""

from setuptools import setup, find_packages

setup(
    name="stackdb",
    version="0.1.0",
    author="Evan Phibbs",
    description="An in-memory vector database with similarity search",
    url="https://github.com/ephibbs/stackdb",
    packages=find_packages(include=["stackdb", "stackdb.*"]),
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.24.3",
        "pydantic>=2.11.5",
        "scikit-learn>=1.3.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=25.1.0",
            "flake8>=6.0.0",
            "mypy>=1.15.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="vector database, similarity search, machine learning, embeddings, cosine similarity",
    project_urls={
        "Bug Reports": "https://github.com/ephibbs/stackdb/issues",
        "Source": "https://github.com/ephibbs/stackdb",
        "Documentation": "https://github.com/ephibbs/stackdb/README.md",
    },
)
