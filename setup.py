from setuptools import find_packages, setup
import codecs

version = "0.0.1"

with open("README.md", "r", encoding="utf-8") as readme_file:
    README = readme_file.read()


def parse_requirements(fname='requirements.txt'):
    """Parse the package dependencies listed in a requirements file."""

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            target = line.split(' ')[1]
            for line in parse_require_file(target):
                yield line
        else:
            yield line

    def parse_require_file(fpath):
        with codecs.open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for ll in parse_line(line):
                        yield ll

    packages = list(parse_require_file(fname))
    return packages

setup(
    name="kantts",
    version=version,
    url="https://github.com/AlibabaResearch/KAN-TTS",
    author="Jin",
    description="Alibaba DAMO Speech-Lab Text to Speech deeplearning toolchain",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    # cython
    #  include_dirs=numpy.get_include(),
    # ext_modules=find_cython_extensions(),
    # package
    include_package_data=True,
    packages=find_packages(include=["kantts*"]),
    project_urls={
        "Documentation": "https://github.com/AlibabaResearch/KAN-TTS/wiki",
        "Tracker": "",
        "Repository": "https://github.com/AlibabaResearch/KAN-TTS",
        "Discussions": "",
    },
    python_requires=">=3.7.0, <3.9",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=parse_requirements('requirements/runtime.txt'),
    zip_safe=False,
)
