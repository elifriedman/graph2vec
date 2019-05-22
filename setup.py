from setuptools import setup, find_packages

setup(
    name='graph2vec',
    version='1.0',
    description='Graph2Vec',
    author='Eli Friedman',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['numpy', 'networkx', 'gensim', 'scipy'],
    scripts=['src/graph2vec.py'],
    extras_require={
        'test': ['pytest'],
    },
)
