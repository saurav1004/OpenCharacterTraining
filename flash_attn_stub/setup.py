from setuptools import setup, find_packages

setup(
    name="flash-attn",
    version="2.8.3-stub",
    description="Python-only stub that satisfies flash-attn imports without CUDA builds. "
                "Intended for environments using --attn_implementation sdpa.",
    packages=find_packages(),
    install_requires=["torch", "einops"],
)
