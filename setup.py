from setuptools import setup
exec(open('rlxml/version.py').read())

setup(name='rlxml',
      version=__version__,
      description='rlx machine learning and stats',
      url='http://github.com/rramosp/rlxml',
      install_requires=['matplotlib','numpy', 'pandas','joblib',
                        'progressbar2', 'psutil', 'bokeh', 'pyshp',
                        'statsmodels', 'scipy'],
      scripts=[],
      author='rlx',
      author_email='rulix.rp@gmail.com',
      license='MIT',
      packages=['rlxml'],
      include_package_data=True,
      zip_safe=False)
