# Oracle Guardian AI Open Source Project

[![PyPI](https://img.shields.io/badge/pypi-v1.0.0-blue?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/oracle-guardian-ai/) [![Python](https://img.shields.io/badge/python-3.8-blue?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/oracle-guardian-ai/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://github.com/ambv/black)

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

To work with fairness/bias, install the `fairness` module. You can find extra dependencies in [requirements-fairness.txt](./guardian_ai/requirements-fairness.txt).

```bash
python3 -m pip install 'oracle-guardian-ai[fairness]'
```

To work with privacy estimation, install the `privacy` module. You can find extra dependencies in [requirements-privacy.txt](./guardian_ai/requirements-privacy.txt).

```bash
python3 -m pip install 'oracle-guardian-ai[privacy]'
```

## Documentation
  - [Oracle Guardian AI Documentation](https://oracle-guardian-ai.readthedocs.io/en/latest/index.html)
  - [OCI Data Science and AI services Examples](https://github.com/oracle/oci-data-science-ai-samples)
  - [Oracle AI & Data Science Blog](https://blogs.oracle.com/ai-and-datascience/)

## Contributing

This project welcomes contributions from the community. Before submitting a pull request, please review our [contribution guide](./CONTRIBUTING.md).

Find Getting Started instructions for developers in [README-development.md](./README-development.md).

## Security

Consult the security guide [SECURITY.md](./SECURITY.md) for our responsible security vulnerability disclosure process.

## License

Copyright (c) 2023 Oracle and/or its affiliates. Licensed under the [Universal Permissive License v1.0](https://oss.oracle.com/licenses/upl/).


