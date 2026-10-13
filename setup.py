from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="all_classifier_666",
    version="0.1.0",
    author="Arı Bilgi",
    author_email="aribilgiogr@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aribilgiogr/Classifier666",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "scikit-learn", "xgboost", "lightgbm"],
    classifiers=[
        "Programming Language :: Python:: 3",
        "License :: OSI Approves :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.10'
)
