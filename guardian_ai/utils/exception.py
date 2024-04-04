#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Exception module."""


class GuardianAIError(Exception):
    """GuardianAIError

    The base exception from which all exceptions raised by GuardianAI
    will inherit.
    """

    pass


class GuardianAIValueError(ValueError, GuardianAIError):
    """Exception raised for unexpected values."""

    pass


class GuardianAITypeError(TypeError, GuardianAIError):
    """Exception raised for generic type issues."""

    pass


class GuardianAIRuntimeError(RuntimeError, GuardianAIError):
    """Exception raised for generic errors at runtime."""

    pass


class GuardianAIImportError(ImportError, GuardianAIError):
    """Exception raised for import errors when lazy loading."""

    pass


class GuardianAINotImplementedError(NotImplementedError, GuardianAIError):
    """Exception raised when accessing code that has not been implemented."""

    pass


class GuardianAIProgrammerError(GuardianAIError):
    """Exception raised for errors related to unexpected implementation issues."""

    pass
