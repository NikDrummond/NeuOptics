from setuptools import setup, find_packages
import re

VERSIONFILE = "__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
	verstr=mo.group(1)
else:
	raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('requirements.txt') as f:
	requirements = f.read().splitlines()
	requirements = [l for l in requirements if not l.startswith('#')]

setup(name='NeuOptics',
	version=verstr,
	description='Neurosetta add-on for optic lobe data in drosophila',
	url='',
	author='Nik Drummond',
	author_email='nikolas.drummond@bi.mpg.de',
	license='MIT',
	packages=find_packages(),
	extras_require={
        "dev": ["pytest", "sphinx", "sphinx-rtd-theme"],
    },
	long_description=open("README.md").read(),
	long_description_content_type="text/markdown",
	zip_safe=False,
	include_package_data = True,
	package_data = {"NeuOptics": [data]}
)