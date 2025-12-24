from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name='steer',
    version='1.5.0',
    packages=find_packages(),
    include_package_data=True,
    description='STEER: Spatial-Temporal Explainable Expert model for RNA velocity inference',
    author='Liu Zhiyuan',  # 替换成你的名字
    author_email='lzy_math@163.com',  # 替换成你的邮箱
    url='https://github.com/lzygenomics/STEER',  # 替换成你的GitHub或项目地址
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',  # 根据实际情况选择许可证
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
)
