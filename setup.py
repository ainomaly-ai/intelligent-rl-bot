from setuptools import setup

with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f]

setup(
    name="RL-trade-bot",
    version="0.0.1",
    install_requires=install_requires,
)
