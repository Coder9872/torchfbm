import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchFBM",
    version="0.1.0",
    author="Ivan Habib",
    author_email="ivan.habib4@gmail.com",
    description="Fractional Brownian motion and related processes in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Coder9872/torchFBM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
