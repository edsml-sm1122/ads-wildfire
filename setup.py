try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='wildfire',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='1.0',
      description='wildfire surrogate models',
      author='ADS Creek',
      packages=['wildfire']
      )
