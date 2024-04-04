***********
Quick Start
***********

This section provides a quick introduction about how to use the ``oracle-guardian-ai`` package.


Installation
============

- Installing the ``oracle-guardian-ai`` base package

  ..  code-block:: shell

    pip install oracle-guardian-ai

- Installing extras libraries

The ``all-optional`` module will install all optional dependencies. Note the single quotes around installation of extra libraries.

  ..  code-block:: shell

    pip install 'oracle-guardian-ai[all-optional]'


To work with fairness/bias, install the ``fairness`` module.

  ..  code-block:: shell

    pip install 'oracle-guardian-ai[fairness]'

To work with privacy estimation, install the ``privacy`` module.

  ..  code-block:: shell

    python3 -m pip install 'oracle-guardian-ai[privacy]'


.. include:: user_guide/fairness/quickstart.rst

.. include:: user_guide/privacy_estimation/quickstart.rst
