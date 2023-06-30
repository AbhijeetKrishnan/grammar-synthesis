from setuptools import setup

with open('README.md') as readme:
    long_description = readme.read()

setup(
    name="grammar_synthesis",
    version="0.0.1",
    author="Abhijeet Krishnan",
    author_email="abhijeet.krishnan@gmail.com",
    description="Gymnasium environment for CFG-based program synthesis.",
    long_description=long_description,
    long_description_context_type="text/markdown",
    url="https://github.com/AbhijeetKrishnan/grammar-synthesis",
    python_requires=">=3.11.3",
    install_requires=["gymnasium>=0.28.1", "lark>=1.1.2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: The Unlicense (Unlicense)",
        "Operating System :: OS Independent",
    ],
)