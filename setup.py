# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()

with open(here/'README.md') as fh:
    long_description = fh.read()

setup(
    name='sudokius',
    version='1.0.0',
    author='Greg Sotiropoulos',
    author_email='greg.sotiropoulos@gmail.com',
    description='A Sudoku puzzle solver',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['sudoku', 'puzzle', 'solver', 'algorithms', 'games'],
    url='https://github.com/gregsotiropoulos/sudoku',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Games/Entertainment :: Puzzle Games',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy'
    ],
    package_data={
        'sudoku17 strings': ['sudoku17.txt'],
        'application icon': ['sudoku.png']
    },
    project_urls={
        'Source': 'https://github.com/gregsotiropoulos/sudoku'
    }
)
