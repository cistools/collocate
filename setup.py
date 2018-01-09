import os.path

from setuptools import setup, find_packages, Command
from pkg_resources import require, DistributionNotFound, VersionConflict
from colocate import __version__


root_path = os.path.dirname(__file__)

# If we're building docs on readthedocs we don't have any dependencies as they're all mocked out
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    dependencies = []
    optional_dependencies = {}
    test_dependencies = []

else:
    dependencies = ["xarray",
                    "scipy>=0.15.0"]
    optional_dependencies = ['iris']
    test_dependencies = []


class check_dep(Command):
    """
    Command to check that the required dependencies are installed on the system
    """
    description = "Checks that the required dependencies are installed on the system"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        for dep in dependencies:
            try:
                require(dep)
                print(dep + " ...[ok]")
            except (DistributionNotFound, VersionConflict):
                print(dep + "... MISSING!")


# Extract long-description from README
README = open(os.path.join(root_path, 'README.md')).read()


setup(
    name='colocate',
    version=__version__,
    description='Colocate',
    long_description=README,
    maintainer='Duncan Watson-Parris',
    maintainer_email='duncan.watson-parris@physics.ox.ac.uk',
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        ],
    packages=find_packages(),
    cmdclass={"checkdep": check_dep},
    install_requires=dependencies,
    extras_require=optional_dependencies,
    tests_require=test_dependencies
)
