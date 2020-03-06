from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
  long_description = fh.read()

with open('requirements.txt', 'r') as fh:
  requirements = fh.read().split('\n')

setup(name='spectra-torch',
      version='0.3.0',
      author='WangRui',
      author_email='rwang@tongji.edu.cn',
      description='Spectra Extraction based on PyTorch',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/mechanicalsea/spectra',
      packages=['spectra_torch'],
      install_requires=requirements,
      classifiers=[
          "Programming Language :: Python :: 3.7",
      ],
)
# python3 setup.py sdist build
# python3 setup.py bdist_wheel --universal
# twine upload dist/*
