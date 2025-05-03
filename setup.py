from setuptools import setup, find_packages
setup(
name="segmentacija-slik",
version="0.1",
packages=find_packages(),
install_requires=[
"numpy>=1.19.0",
"opencv-python>=4.5.0",
"matplotlib>=3.3.0",
],
)