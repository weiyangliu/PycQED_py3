# This is a PowerShell based script used to install PycQED on your Windows OS computer.

# Prerequisite:
#  - Anaconda based on Python 3.5+
#    - Download: https://www.continuum.io/downloads
#  - The PyqCED source code
#    - Download: https://github.com/DiCarloLab-Delft/PycQED_py3

conda create -n pycqed python=3

# if working properly, this step will install all required packages automatically.
python -m pip install --upgrade pip

# qcodes should be installed from DiCarloLab-Delft. Downloading is required.
pip install qcodes

# install PyqCED
python setup.py develop
