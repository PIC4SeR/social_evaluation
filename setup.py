import os

from setuptools import find_packages, setup

with open(os.path.join("social_evaluation", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

long_description = """

# Social evaluation

This repository some utilities to evaluate the social behavior of controllers 
from a set of rosbags. 

## Links

Repository:
https://github.com/PIC4SeR/social-evaluation


"""
setup(
    name="social-evaluation",
    packages=[
        package
        for package in find_packages()
        if package.startswith("social_evaluation")
    ],
    package_data={"social_evaluation": ["py.typed", "version.txt"]},
    install_requires=[
        "numpy",
        # For reading logs. Keep pandas below 2.2 and matplotlib at
        # least 3.6.3 to avoid the Series indexing failure triggered
        # by seaborn's categorical plots with older matplotlib.
        "pandas>=1.5,<2.2",
        # Plotting stack used by benchmark_report.
        "matplotlib>=3.6.3,<3.9",
        "seaborn>=0.13,<0.14",
        # for reading rosbags
        # "rosbag2_py",
        # "nav_msgs",
        # "people_msgs",
        # "tf2_msgs",
        # "sensor_msgs",
        "rliable",
        "argcomplete",
        "pyyaml",
        "argparse",
        "tf-transformations",
        "transforms3d>=0.4.1",
    ],
    entry_points={
        "console_scripts": [
            "bag_extraction=social_evaluation.bag_extraction_:main",
            "data_filtering=social_evaluation.data_filtering_:main",
            "plot_trajectories=social_evaluation.plot_trajectories_:main",
            "metrics_evaluation=social_evaluation.metrics_evaluation_:main",
            "benchmark_report=social_evaluation.benchmark_report_:main",
        ],
    },
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "mypy",
            # Lint code and sort imports (flake8 and isort replacement)
            "ruff>=0.3.1",
            # Reformat
            "black>=24.2.0,<25",
        ],
        # "docs": [
        #     "sphinx>=5,<8",
        #     "sphinx-autobuild",
        #     "sphinx-rtd-theme>=1.3.0",
        #     # For spelling
        #     "sphinxcontrib.spelling",
        #     # Copy button for code snippets
        #     "sphinx_copybutton",
        # ],
    },
    description="Social Evaluation of Robot Local Planners Algorithms",
    author="Andrea Ostuni",
    url="https://github.com/PIC4SeR/social-evaluation",
    author_email="andrea.ostuni@polito.it",
    keywords="social-navigation robot local-planner ros2",
    license="GNU General Public License v3.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.8",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/PIC4SeR/social-evaluation",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
