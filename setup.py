from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='suspicious',
    version='0.1.1',
    author='Kiril Videlov',
    author_email='kiril@codeball.ai',
    description='Detects possibly suspicious stuff in your source files',
    install_requires=[
        'Jinja2==3.1.2',
        'numpy==1.22.4',
        'setuptools==65.5.1',
        'torch==1.12.1',
        'tqdm==4.64.0',
        'transformers==4.24.0',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    url='https://github.com/sturdy-dev/suspicious',
    packages=find_packages(where="src"),
    package_dir={
        "": "src"
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'sus=suspicious.cli:main',
        ]
    },
    keywords='code analysis',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development',
        'Topic :: Utilities',
    ]
)
