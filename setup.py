from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name='steer',
    version='2.2.1',
    packages=find_packages(),
    include_package_data=True,
    description='STEER: Spatial-Temporal Explainable Expert model for RNA velocity inference',
    author='Liu Zhiyuan',
    author_email='lzy_math@163.com',
    url='https://github.com/lzygenomics/STEER',
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
)
