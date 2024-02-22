import os.path as osp
from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        content = f.read()
    return content



def get_requirements(filename='requirements.txt'):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


setup(
    name='reid',
    version=0,
    description='Computer Science graduation project on person re-identification.',
    author='Selin Kayay',
    license='MIT',
    long_description=readme(),
    packages=find_packages(),
    install_requires=get_requirements(),
    keywords=['Person Re-Identification', 'Deep Learning', 'Computer Vision']
)