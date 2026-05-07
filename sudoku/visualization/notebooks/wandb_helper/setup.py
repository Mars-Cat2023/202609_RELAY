from setuptools import setup

setup(
    name="wandb_helper",
    version="0.1.0",
    description="Utilities for fetching and processing Weights & Biases runs",
    packages=["wandb_helper"],
    package_dir={"wandb_helper": "."},
    package_data={"wandb_helper": ["configs/*.yaml", "configs/*.yml"]},
    include_package_data=True,
    install_requires=[
        "pandas",
        "wandb",
        "hydra-core",
    ],
    python_requires=">=3.8",
)
