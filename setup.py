from setuptools import setup, find_packages

setup(
    name="parq",
    version="0.1.0",
    description="A Python package for efficient handling of Parquet datasets.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="bn999",
    author_email="your.email@example.com",
    url="https://github.com/bn999/parq",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0",
        "pyarrow>=8.0",
    ],  # Dependencies required for the package
    python_requires=">=3.6",  # Minimum Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
