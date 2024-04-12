from setuptools import setup, find_packages

setup(
    name='SECAR_NN_Helper',
    version='1.0.0',
    description='Library for the Neural Network model for the FRIB',
    author='Juan José Lozano González',
    author_email='lozanoju@msu.edu',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'seaborn',
        'matplotlib',
        'pandas',
    ],
)