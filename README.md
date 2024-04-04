# Oracle Guardian AI Open Source Project

[![PyPI](https://img.shields.io/pypi/v/oracle-guardian-ai.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/oracle-guardian-ai/) [![Python](https://img.shields.io/pypi/pyversions/oracle-guardian-ai.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/oracle-guardian-ai/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://github.com/ambv/black)

Oracle Guardian AI Open Source Project is a library consisting of tools to assess fairness/bias and privacy of machine learning models and data sets. This package contains `fairness` and `privacy_estimation` modules.

The `Fairness` module offers tools to help you diagnose and understand the unintended bias present in your dataset and model so that you can make steps towards more inclusive and fair applications of machine learning.

The `Privacy Estimation` module helps estimate potential leakage of sensitive information in the training data through attacks on Machine Learning (ML) models. The main idea is to carry out Membership Inference Attacks on a given target model trained on a given sensitive dataset, and measure their success to estimate the risk of leakage.

## Installation

You have various options when installing `oracle-guardian-ai`.

### Installing the oracle-guardian-ai base package

```bash
python3 -m pip install oracle-guardian-ai
```

### Installing extras libraries

The `all-optional` module will install all optional dependencies. Note the single quotes around installation of extra libraries.

```bash
python3 -m pip install 'oracle-guardian-ai[all-optional]'
```

To work with fairness/bias, install the `fairness` module. You can find extra dependencies in [requirements-fairness.txt](https://github.com/oracle/guardian-ai/blob/main/guardian_ai/requirements-fairness.txt).

```bash
python3 -m pip install 'oracle-guardian-ai[fairness]'
```

To work with privacy estimation, install the `privacy` module. You can find extra dependencies in [requirements-privacy.txt](https://github.com/oracle/guardian-ai/blob/main/guardian_ai/requirements-privacy.txt).

```bash
python3 -m pip install 'oracle-guardian-ai[privacy]'
```

## Documentation
  - [Oracle Guardian AI Documentation](https://oracle-guardian-ai.readthedocs.io/en/latest/index.html)
  - [OCI Data Science and AI services Examples](https://github.com/oracle/oci-data-science-ai-samples)
  - [Oracle AI & Data Science Blog](https://blogs.oracle.com/ai-and-datascience/)

## Examples

### Measurement with a Fairness Metric

```python
from guardian_ai.fairness.metrics import ModelStatisticalParityScorer
fairness_score = ModelStatisticalParityScorer(protected_attributes='<target_attribute>')
```

### Bias Mitigation

```python
from guardian_ai.fairness.bias_mitigation import ModelBiasMitigator
bias_mitigated_model = ModelBiasMitigator(
    model,
    protected_attribute_names='<target_attribute>',
    fairness_metric="statistical_parity",
    accuracy_metric="balanced_accuracy",
)

bias_mitigated_model.fit(X_val, y_val)
bias_mitigated_model.predict(X_test)
```


## Contributing

This project welcomes contributions from the community. Before submitting a pull request, please review our [contribution guide](./CONTRIBUTING.md).

Find Getting Started instructions for developers in [README-development.md](https://github.com/oracle/guardian-ai/blob/main/README-development.md).

## Security

Consult the security guide [SECURITY.md](https://github.com/oracle/guardian-ai/blob/main/SECURITY.md) for our responsible security vulnerability disclosure process.

## License

Copyright (c) 2023 Oracle and/or its affiliates. Licensed under the [Universal Permissive License v1.0](https://oss.oracle.com/licenses/upl/).
