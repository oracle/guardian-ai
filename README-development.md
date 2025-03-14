# Development
The target audience for this README is developers wanting to contribute to `oracle-guardian-ai`. If you want to use the Oracle Guardian AI Open Source Project with your own programs, see `README.md`.

## Get Support

- Open a [GitHub issue](https://github.com/oracle/guardian-ai/issues) for bug reports, questions, or requests for enhancements.
- Report a security vulnerability according to the [Reporting Vulnerabilities guide](https://www.oracle.com/corporate/security-practices/assurance/vulnerability/reporting.html).


## Setting Up Dependencies

These are the minimum required steps to install and set up the Oracle Guardian AI Project to run on your local machine
for development and testing purposes.
### Step 1: Create a conda environment

Install Anaconda from `https://repo.continuum.io/miniconda/` for the operating system you are using.

In the terminal client, enter the following where <yourenvname> is the name you want to call your environment,
and set the Python version you want to use. Oracle Guardian AI Project requires Python >=3.9.

```bash
    conda create -n <yourenvname> python=3.9 anaconda
```


This installs the Python version and all the associated anaconda packaged libraries at `path_to_your_anaconda_location/anaconda/envs/<yourenvname>`

### Step 2: Activate your environment

To activate or switch into your conda environment, run this command:

```bash
    conda activate <yourenvname>
```

### Step 3: Clone and install dependencies

Open the destination folder where you want to clone this project, and install dependencies like this:

```bash
    cd <desctination_folder>
    git clone git@github.com:oracle/guardian-ai.git
    python3 -m pip install -r dev-requirements.txt
```



# Running Tests
The SDK uses pytest as its test framework. To run tests use:

```
python3 -m pytest tests/*
```

# Generating Documentation
Sphinx is used for documentation. You can generate HTML locally with the following:

```
python3 -m pip install -r dev-requirements.txt
cd docs
make html
```

# Versioning and generation the wheel

Bump the versions in `pyproject.toml`. The Oracle Guardian AI Project using [build](https://pypa-build.readthedocs.io/en/stable/index.html) as build frontend. To generate sdist and wheel, you can run:

```
pip install build
```

The Oracle Guardian AI are packaged as a wheel. To generate the wheel, you can run:

```
make dist
```

This wheel can then be installed using `pip`.

# Security

Consult the [security guide](https://github.com/oracle/guardian-ai/blob/main/SECURITY.md) for our responsible security vulnerability disclosure process.

# License

Copyright (c) 2023 Oracle, Inc. All rights reserved.
Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl.
