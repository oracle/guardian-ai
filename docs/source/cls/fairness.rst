.. _fairness_cls:

********
Fairness
********


.. automodule:: guardian_ai.fairness


Metrics
=======

.. automodule:: guardian_ai.fairness.metrics

Evaluating a Model
------------------

Statistical Parity
^^^^^^^^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.model.ModelStatisticalParityScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.model.model_statistical_parity

True Positive Rate Disparity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.model.TruePositiveRateScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.model.true_positive_rate

False Positive Rate Disparity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.model.FalsePositiveRateScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.model.false_positive_rate


False Negative Rate Disparity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.model.FalseNegativeRateScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.model.false_negative_rate

False Omission Rate Disparity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.model.FalseOmissionRateScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.model.false_omission_rate

False Discovery Rate Disparity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.model.FalseDiscoveryRateScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.model.false_discovery_rate

Error Rate Disparity
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.model.ErrorRateScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.model.error_rate

Equalized Odds
^^^^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.model.EqualizedOddsScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.model.equalized_odds

Theil Index
^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.model.TheilIndexScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.model.theil_index

Evaluating a Dataset
--------------------

Statistical Parity
^^^^^^^^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.dataset.DatasetStatisticalParityScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.dataset.dataset_statistical_parity

Consistency
^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.dataset.ConsistencyScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.dataset.consistency


Smoothed EDF
^^^^^^^^^^^^

.. autoclass:: guardian_ai.fairness.metrics.dataset.SmoothedEDFScorer
    :members:
    :inherited-members:
    :special-members: __call__

.. autofunction:: guardian_ai.fairness.metrics.dataset.smoothed_edf


Bias Mitigation
===============

.. automodule:: guardian_ai.fairness.bias_mitigation

Bias Mitigator
--------------


.. autoclass:: guardian_ai.fairness.bias_mitigation.sklearn.ModelBiasMitigator
    :members:
    :inherited-members:
    :special-members: __call__
