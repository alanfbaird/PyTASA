try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Python Tools for Analysing Seismic Anisotropy',
    'version': '0.1.0',
    'install_requires': ['nose'],
    'packages': ['pytasa'],
    'name': 'PyTASA'
}

setup(**config)
