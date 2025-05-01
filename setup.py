from setuptools import setup, find_packages

setup(
    name="deepdig",
    version="0.1",
    packages=find_packages(),
    package_dir={'': '.'},  # Important for flat structure
)
