#######################################
Oracle Guardian AI Open Source Project
#######################################

Oracle Guardian AI Open Source Project is a library consisting of tools to assess fairness/bias and privacy of machine learning models and data sets.
This package contains ``fairness`` and ``privacy_estimation`` modules.

The :ref:`Fairness module <fairness_cls>` offers tools to help you diagnose and understand the unintended bias present in your
dataset and model so that you can make steps towards more inclusive and fair applications of machine learning.

The :ref:`Privacy Estimation module <privacy_cls>` helps estimate potential leakage of sensitive information in the training
data through attacks on Machine Learning (ML) models. The main idea is to carry out Membership Inference Attacks on a given
target model trained on a given sensitive dataset, and measure their success to estimate the risk of leakage.

Getting Started
===============
Head to :doc:`quickstart` to see how you can get started with ``oracle-guardian-ai``.



.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Getting Started:

    quickstart
    release_notes
    user_guide/fairness/overview
    user_guide/privacy_estimation/privacy

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Class Documentation:

    cls/fairness
    cls/privacy
