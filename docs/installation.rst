.. _picasso_addon:
	https://github.com/schwille-paint/picasso_addon
.. _SPT:
	https://github.com/schwille-paint/SPT

Installation
============

.. toctree::
   :maxdepth: 2

Create conda environment   
^^^^^^^^^^^^^^^^^^^^^^^^
Since the SPT package is based on the `picasso_addon`_ package, please follow the instructions 
`how to set up a conda environment for picasso_addon <https://picasso-addon.readthedocs.io/en/latest/installation.html>`_. The thus created environment (``picasso_addon``) provides all
necessary dependencies for SPT.

Download and use SPT
^^^^^^^^^^^^^^^^^^^^
To use the SPT package please clone the `SPT`_ GitHub repository. 
You can add the package to your environment (e.g. ``picasso_addon``) by switching to the downloaded folder (SPT) and typing

.. code-block:: console

    (picasso_addon) python setup.py install

 
If you don't want to install the SPT package into your environment but want to be able to permanently import SPT functions in any IPython (Spyder, Jupyter) session do the following:
    1. Navigate to ~/.ipython/profile_default
    2. Create a folder called startup if it’s not already there
    3. Add a new Python file called ``start.py`` or modify it and add 
    
    .. code-block:: python

        import sys
        sys.path.append('C:/yourpath/SPT')

 