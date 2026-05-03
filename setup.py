from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(
    name="iris-flower-classification",
    version="0.1.0",
    author="Vighanesh Thakare",
    author_email="thakarevighanesh@gmail.com",
    description="Flask-based Iris Flower Classification using Deep Learning",
    long_description="A machine learning project that predicts Iris flower species using a neural network and provides a web interface using Flask.",
    long_description_content_type="text/plain",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.12",
)