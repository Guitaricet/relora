from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="peft_pretraining",
    version="1.0",
    description="ReLoRA: Parameter-efficient pre-training",
    url="https://github.com/Guitaricet/peft_pretraining",
    author="Vlad Lialin",
    author_email="vlad.lialin@gmail.com",
    license="Apache 2.0",
    packages=["peft_pretraining"],
    install_requires=required,
)
