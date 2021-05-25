from setuptools import setup

setup(
   name='SPT',
   version='1.0.0',
   description='Package to analyze single-particle-tracking data for immobilized and mobile case',
   license="MIT License",
   author='Stehr Florian',
   author_email='florian.stehr@gmail.com',
   url="http://www.github.com/schwille-paint/SPT",
   packages=['spt'],
   classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
   ],
   install_requires=[
		"picasso @ git+https://github.com/jungmannlab/picasso.git#egg=picasso-0.3.1",
		"picasso_addon @ git+https://github.com/schwille-paint/picasso_addon.git#egg=picasso_addon-0.1.1",
   ], 
)
