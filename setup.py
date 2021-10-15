import setuptools

install_requires = [
]

docs_require = [
    'sphinx >= 1.4',
    'sphinx-rtd-theme'
]

setuptools.setup(
    name='libqif',
    version='1.0',
    description='Python library for Quantitative Information Flow (QIF)',
    author='Ramon Gonçalves Gonze',
    author_email='ramongonze@gmail.com',
    url='https://github.com/ramongonze/libqif',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={'docs': docs_require},
    python_requires='>=3.8'
)
