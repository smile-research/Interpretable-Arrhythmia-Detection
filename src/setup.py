# -*- coding: utf-8 -*-
from setuptools import setup, find_namespace_packages

package_data = {"": ["*"]}

dev_requirements = ["pytorch", "pytorch_lightning", "scikit-learn"]


setup(
    name="ecg_benchmarking_lit",
    version="0.1.0",
    packages=find_namespace_packages(),
    package_data=package_data,
    python_requires=">=3.8.0,<4.0.0",
    extras_require={"dev": dev_requirements},
)
