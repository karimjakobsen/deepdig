from setuptools import setup, find_packages

setup(
    name='deepdig',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',  # Add other dependencies if needed
    ],
    author='Karim Jakobsen',
    description='Minimalist deep learning framework',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
