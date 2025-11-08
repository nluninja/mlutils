from setuptools import setup, find_packages

setup(
    name="dvtm-utils",  # Unique name on PyPI
    version="0.1.0",  # Follow semantic versioning
    author="Andrea Belli",
    author_email="andrea.belli@gmail.com",
    description="Utils package for loading datasets and models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nluninja/mlutils",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=['numpy', 'scikit-learn','pandas', 'tensorflow'],
)
