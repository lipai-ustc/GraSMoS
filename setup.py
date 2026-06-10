from setuptools import setup, find_packages

setup(
    name='grasmos',
    version='0.2.0',
    license='GPL-3.0-or-later',
    packages=find_packages(),
    py_modules=['cosmos_run', 'cosmos_search', 'cosmos_utils'],
    entry_points={
        'console_scripts': [
            'grasmos = cosmos_run:main',
        ],
    },
    install_requires=[
        'ase>=3.26.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'dscribe>=2.1.0',
        'networkx>=2.6',
    ],
    author='GraSMoS Development Team',
    author_email='grasmos@example.com',
    description='GraSMoS: Graph-Aware Structure Search with Monte Carlo Simulation',
    long_description=open('ReadMe.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/grasmos',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)