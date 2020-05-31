import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bayesian_linear_regression",
    version="0.0.1",
    author="Marko Djordjic",
    author_email="marko.djordjic@outlook.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markodjordjic",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 2-Clause 'Simplified' License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)