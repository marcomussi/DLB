**********
DynLin-UCB
**********

Marco Mussi, Alberto Maria Metelli and Marcello Restelli

Running Experiments
===================

The code requires *python3* and the following packages: *numpy tqdm matplotlib(v3.5.4) tikzplotlib(v0.10.1)*

Experiments script is located in the *scripts* folder.

The configurations used in the experiments of both the main paper and the supplementary material are in the *config* folder.

To run the experiments of the main paper type (from the root directory):
> python3 scripts/main.py <config_filename_without_extension>

Cite DynLin-UCB
===============
If you are using this code for your scientific publications, please cite:

.. code:: bibtex

    @inproceedings{mussi2023dynamical,
      author    = {Mussi, Marco and Metelli, Alberto Maria and Restelli, Marcello},
      title        = {Dynamical Linear Bandits},
      booktitle    = {International Conference on Machine Learning (ICML)},
      series       = {Proceedings of Machine Learning Research},
      volume       = {202},
      pages        = {25563--25587},
      publisher    = {{PMLR}},
      year         = {2023}
   }

How to contact us
=================
For any question, drop an e-mail at marco.mussi@polimi.it
