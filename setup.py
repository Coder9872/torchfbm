import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchfbm",
    version="0.2.1",
    author="Ivan Habib",
    author_email="ivan.habib4@gmail.com",
    description="High-performance Fractional Brownian Motion toolkit for PyTorch with generators, processes, neural layers, and RL support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/i-habib/torchfbm",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
        ],
        "rl": [
            "stable-baselines3>=1.0",
            "gym>=0.18.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Coder9872/torchfbm/issues",
        "Source": "https://github.com/Coder9872/torchfbm",
        "Documentation": "https://github.com/Coder9872/torchfbm/blob/main/README.md",
    },
)
