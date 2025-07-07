from setuptools import setup, find_packages

setup(
    name="trading-stock-agents",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "pytest",
    ],
    python_requires=">=3.11",
)
