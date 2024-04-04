********************************
Unintended Bias and Fairness
********************************

Protected attributes are referred to as
features that may not be used as the basis for decisions (for example,
race, gender, etc.). When machine learning is applied to decision-making
processes involving humans, one should not only look for models with
good performance, but also models that do not discriminate against
protected population subgroups.

Oracle Guardian AI Project provides metrics dedicated to assessing and measuring the
compliance of a model or a dataset with a fairness metric. The provided
metrics all correspond to different notions of fairness, which the user
should carefully select from while taking into account their problem’s
specificities.

It also provides a bias mitigation algorithm that fine-tunes
decison thresholds across demographic groups to compensate for the bias
present in the original model. The approach is called Bias Mitigation.

.. toctree::
    :maxdepth: 3

    fairness_metrics
    fairness_bias_mitigation
