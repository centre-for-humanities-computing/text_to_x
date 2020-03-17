from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='normalize_text_to_df',
   version='1.0',
   description='A pipeline for normalizing texts to dataframes',
   license='MIT',
   long_description=long_description,
   long_description_content_type="text/markdown",
   author='Kenneth C. Enevoldsen',
   author_email='kennethcenevoldsen@gmail.com',
   url="https://github.com/centre-for-humanities-computing/normalize_text_to_df",
   packages=['normalize_text_to_df'],  #same as name
   install_requires=['numpy', 'pandas', 'stanfordnlp', 'polyglot'], #external packages as dependencies

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='Natural Language Processing',
)