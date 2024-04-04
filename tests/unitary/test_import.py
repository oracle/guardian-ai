#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from guardian_ai import *


def test_import():
    import guardian_ai

    from guardian_ai import fairness
    from guardian_ai import privacy_estimation

    assert True
