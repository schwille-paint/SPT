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
   dependency_links=[
		"https://github.com/jungmannlab/picasso/tarball/master",
		"https://github.com/schwille-paint/picasso_addon/tarball/master",
   ], 
)
