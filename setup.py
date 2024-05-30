from setuptools import setup, find_packages

requirements = [
    "h5py>=3.8.0",
    "numpy>=1.23",
    "torch>=2.2",
    "torch_scatter>=2.1.2",
    "ase>=3.22.0",
    "tqdm>=4.65",
]

setup(
    name="radnet",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
)
