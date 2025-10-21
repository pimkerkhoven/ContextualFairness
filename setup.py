# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.


import setuptools

import contextualfairness

with open("README.md", "r") as fh:
    long_description = fh.read()

# with open("requirements.txt", "r") as f:
#     install_requires = [line.strip() for line in f]

install_requires = ["pandas==2.3.3", "numpy==2.2.6"]

setuptools.setup(
    name=contextualfairness.__name__,
    version=contextualfairness.__version__,
    author=("Pim Kerkhoven"),
    author_email="pimk@cs.umu.se",
    description="A python package for assessing machine learning fairness with multiple contextual norms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pimkerkhoven/ContextualFairness",
    packages=setuptools.find_packages(),
    python_requires=">=3.10.13",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
