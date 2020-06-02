.. _picasso_addon:
	https://github.com/schwille-paint/picasso_addon
.. _picasso:
	https://github.com/jungmannlab/picasso
.. _picasso.localize:
	https://picassosr.readthedocs.io/en/latest/localize.html
.. _trackpy:
	http://soft-matter.github.io/trackpy/v0.4.2/

SPT: Single particle tracking analysis
======================================

This package provides a complete single particle tracking analysis workflow based on `picasso_addon`_, `picasso`_ and `trackpy`_ python packages including:

- Localization of raw movies based on `picasso_addon`_ (auto net-gradient) and `picasso.localize`_
- Autopicking of localization clusters (`picasso_addon`_) and analysis of immobilized particles
- Linking of localizations into trajectories using `trackpy`_
- Individual mean-square-displacement computation and linear iterative fitting
- Subtrajectory analysis for estimation of underlying diffusion behavior
- Easy to use script batch processing

SPT requires the following packages:
    - `picasso`_  :  Localization and rendering of super-reolution images
    - `picasso_addon`_ : Further functionalities for picasso (auto net-gradient, autopick)

.. image:: files/software-immob.png
    :width: 600px
    :align: center
    :alt: Workflow

SPT was used for data analysis in:

- `Tracking Single Particles for Hours via Continuous DNA-mediated Fluorophore Exchange <https://www.biorxiv.org/content/10.1101/2020.05.17.100354v1>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents
   
   installation
   howto
   modules
   contact

