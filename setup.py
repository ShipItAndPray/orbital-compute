from setuptools import setup, find_packages

setup(
    name="orbital-compute",
    version="0.1.0",
    description="Simulate and schedule compute jobs across satellite constellations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ShipItAndPray",
    url="https://github.com/ShipItAndPray/orbital-compute",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "sgp4>=2.22",
        "numpy>=1.24",
    ],
    entry_points={
        "console_scripts": [
            "orbital-sim=run_sim:main",
            "orbital-dashboard=dashboard:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: System :: Distributed Computing",
    ],
)
