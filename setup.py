from setuptools import setup, find_packages

__version__ = "0.1"


setup(name='parkloader',
      version=__version__,
      description='Loader for the parkinsons disease patients gait and handwriting data.',
      url='https://github.com/patientzero/parkloader.git',
      packages=find_packages(),
      install_requires=["pandas"]
      )
