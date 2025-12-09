
Conda Environment MLEnv
python = 3.10.18

A setup.py file is required when you want to turn your Python project into an installable Python package.
It is the traditional (and still widely used) way to:

✅ 1. Install your project using:
pip install .


or

pip install -e .


Without setup.py, pip doesn’t know:

What your package is named

What version it is

What dependencies it requires


-e . in requirements.txt file ?
Editable mode installs your project so that any changes you make to your source code are immediately reflected, without reinstalling the package.

This is extremely useful in machine learning or modular projects where you have a folder like:

once the project package is created it will create a egg-info file is created and which confirms that proejct is packaged rightly.