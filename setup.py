from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys
import pathlib

class CustomInstallCommand(install):
    def run(self):
        install.run(self)

        # Determine the script to run based on the operating system
        if sys.platform.startswith('linux') or sys.platform == 'darwin':
            script_path = "scripts/install.sh"
            subprocess.check_call(['chmod', '+x', script_path])
            subprocess.check_call([script_path])
        elif sys.platform == "win32":
            script_path = "scripts/install.bat"
            subprocess.check_call([script_path])
        else:
            raise Exception("Unsupported OS")

setup(
    name='research_analytics',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "seaborn",
        "beautifulsoup4",
        "gspread",
        "oauth2client",
        "lxml",
        "wget",
        "bibtexparser",
        "sentence-transformers",
        "faiss-gpu",
        "flask",
    ],
    cmdclass={
        "install": CustomInstallCommand
    }
)