from setuptools import setup, find_packages

setup(
    name='PySimpleML',
    author='Teja Narasimha Rao',
    author_email='meruvateja123@gmail.com',
    description='A simple Machine Learning built from scratch.'
    long_description=open('README.md').read(),
    long_description_content_type='text/Markdown',
    url='https://github.com/TejMeruva/PySimpleML.git',
    packages=find_packages(),
    requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change if not MIT
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="Machine Learning, Decision Trees, Random Forests, Neural Nrtworks, K Nearest Neighbours, Python"

)