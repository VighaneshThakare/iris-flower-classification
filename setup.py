<<<<<<< HEAD
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
=======
from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='mlproject',
version='0.0.1',
author='Harsh',
author_email='harsh6355@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
>>>>>>> d743ec0794036bbb46ae63f1cece06729a8a3539
)