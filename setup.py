import setuptools


with open('README.md', 'r') as fin:
    long_description = fin.read()


setuptools.setup(
    name='smarttvleakage',
    version='0.1',
    author='Tejas Kannan',
    email='tkannan@uchicago.edu',
    description='Information Leakage on Smart TVs with Audio Signals',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['smarttvleakage'],
    install_requires=[]
)
