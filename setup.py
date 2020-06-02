from setuptools import setup

setup(
   name='SPT',
   version='0.1.1',
   description='Package to analyze single-particle-tracking data for immobilized and mobile case',
   license="Max Planck Institute of Biochemistry",
   author='Stehr Florian',
   author_email='stehr@biochem.mpg.de',
   url="http://www.github.com/schwille-paint/SPT",
   packages=['spt'],
   classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
   ],
   install_requires=[
		"picasso @ git+https://github.com/jungmannlab/picasso.git#egg=picasso-0.3.0",
		"picasso_addon @ git+https://github.com/schwille-paint/picasso_addon.git#egg=picasso_addon-0.1.1",
   ], 
)
