import os, pathlib
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.sdist import sdist
from setuptools import setup, find_packages
from subprocess import check_call

# Package
HERE = pathlib.Path(__file__).parent
PACKAGE_NAME  = 'pulsar_data_collection'
VERSION = '0.0.1'
AUTHOR = 'Rocket Science'
AUTHOR_EMAIL = ''
# URL = 'https://github.com/Rocket-Science-Development/apollo'

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'Pulsar Data Collection is the Model Performance and Management software solution developed by Rocket Science'
LONG_DESCRIPTION = (HERE/"README.md").read_text()
LONG_DESC_TYPE = 'text/markdown'


INSTALL_REQUIRES = [
	"pydantic",
	"pandas",
	"numpy"
]


def gitcmd_update_submodules():
	'''	Check if the package is being deployed as a git repository. If so, recursively
		update all dependencies.

		@returns True if the package is a git repository and the modules were updated.
			False otherwise.
	'''
	if os.path.exists(os.path.join(HERE, '.git')):
		check_call(['git', 'submodule', 'update', '--init', '--recursive'])
		return True

	return False


class gitcmd_develop(develop):
	'''	Specialized packaging class that runs git submodule update --init --recursive
		as part of the update/install procedure.
	'''
	def run(self):
		gitcmd_update_submodules()
		develop.run(self)


class gitcmd_install(install):
	'''	Specialized packaging class that runs git submodule update --init --recursive
		as part of the update/install procedure.
	'''
	def run(self):
		gitcmd_update_submodules()
		install.run(self)


class gitcmd_sdist(sdist):
	'''	Specialized packaging class that runs git submodule update --init --recursive
		as part of the update/install procedure;.
	'''
	def run(self):
		gitcmd_update_submodules()
		sdist.run(self)


setup(cmdclass={
		'develop': gitcmd_develop, 'install': gitcmd_install, 'sdist': gitcmd_sdist,
	}, name=PACKAGE_NAME, version=VERSION, description=DESCRIPTION,
	long_description_content_type=LONG_DESC_TYPE, author=AUTHOR, license=LICENSE, author_email=AUTHOR_EMAIL,
	# url=URL,
	install_requires=INSTALL_REQUIRES,
	packages=find_packages())
