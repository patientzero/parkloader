from setuptools import setup, find_packages

__version__ = "0.1"


setup(name='parkloader',
      version=__version__,
      description='Loader for the parkinsons desease patients gait and handwirting data.',
      url='https://github.com/walwe/ucrloader',
      packages=find_packages(),
      install_requires=["pandas"]
      )
