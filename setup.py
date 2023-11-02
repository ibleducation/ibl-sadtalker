from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = list(set(f.readlines()))

setup(
    name="ibl-sadtalker",
    version="0.0.1",
    description="A lip sync generation library based on github.com/openTalker/SadTalker",
    url="https://github.com/ibleducation/ibl-sadtalker",
    author="IBL",
    author_email="na",
    license="BSD License.",
    include_package_data=True,
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False,
)
