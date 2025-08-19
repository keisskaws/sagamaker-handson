from setuptools import setup, find_packages

setup(
    name="custom-ml-lib",
    version="1.0.0",
    description="カスタム機械学習ライブラリ",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0"
    ],
    python_requires=">=3.7",
)
