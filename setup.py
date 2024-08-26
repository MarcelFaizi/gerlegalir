from setuptools import setup, find_packages

setup(
    name="gerlegalir",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        'numpy',
        'pandas',
        'torch',
        'transformers',
        'sentence-transformers',
        'rank_bm25',
        'pymongo',
        'tqdm',
    ],
)