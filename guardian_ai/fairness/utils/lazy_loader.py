#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Class to lazily load modules."""

import glob
import importlib
import os
from typing import Dict, List, Optional, cast

import pkg_resources  # type: ignore

from guardian_ai.utils.exception import (
    GuardianAIImportError,
    GuardianAIProgrammerError,
    GuardianAIRuntimeError,
)

# Until we find a way to directly parse the config, it is safer to keep it as a global dict
__PARTITIONS__: Optional[Dict[str, List[str]]] = None


def _get_partitions():
    global __PARTITIONS__
    if __PARTITIONS__ is None:
        __PARTITIONS__ = {}
        req_files = glob.glob(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../requirements-*"
            ),
            recursive=True,
        )
        for file in req_files:
            with open(file, "r") as f:
                lines = f.readlines()
                partition_name = file.split("requirements-")[-1].split(".")[0]
                __PARTITIONS__[partition_name] = []
                for line in lines:
                    requirement_name = line.split("==")[0].split("[")[0]
                    __PARTITIONS__[partition_name].append(requirement_name)
        all = []
        for _, deps in __PARTITIONS__.items():
            all += deps
        __PARTITIONS__["all"] = all


# Maps aliases to the corresponding name in __PARTITIONS__
__ALIASES__ = {"sklearn": "scikit-learn", "category_encoders": "category-encoders"}


class LazyLoader:
    """
    Lazy module Loader.
    This object loads a module only when we fetch attributes from it.
    It can be used to import modules in one files which are not
    present in all the runtime environment where
    it will be executed.

    Parameters
    ----------
    lib_name : str
        Full module path (e.g torch.data.utils)

    callable_name : str or None, default=None
        If not ``None``. The Lazy loader only imports a specific
        callable (class or function) from the module

    suppress_import_warnings : bool, default=False
        If True, the import warnings of the package will be
        ignored and removed from output.
    """

    def __init__(
        self,
        lib_name: str,
        callable_name: Optional[str] = None,
        suppress_import_warnings: bool = False,
    ):
        self.lib_name = lib_name
        self._mod = None
        self.callable_name = callable_name
        self.suppress_import_warnings = suppress_import_warnings

    def __load_module(self):
        if self._mod is None:
            if self.suppress_import_warnings:
                import logging

                previous_level = logging.root.manager.disable
                logging.disable(logging.WARNING)
            try:
                self._mod = importlib.import_module(self.lib_name)
                if self.callable_name is not None:
                    self._mod = getattr(self._mod, self.callable_name)
            except ModuleNotFoundError:
                parent_partitions = self._find_missing_partition()
                if len(parent_partitions) > 0:
                    raise GuardianAIImportError(
                        f"Package {self.lib_name.split('.')[0]} is not installed. "
                        f"It is in the following guardian_ai installation options: {parent_partitions}."
                        "Please install the appropriate option for your use case "
                        "with `pip install guardian_ai[option-name]`."
                    )
                else:
                    raise GuardianAIProgrammerError(
                        f"Package {self.lib_name.split('.')[0]} is being lazily loaded "
                        "but does not belong to any partition."
                    )
            finally:
                if self.suppress_import_warnings:
                    logging.disable(previous_level)

    def _find_missing_partition(self):
        _get_partitions()
        global __PARTITIONS__
        parent_partitions = []
        for partition, deps in __PARTITIONS__.items():
            if self.lib_name.split(".")[0] in deps:
                parent_partitions.append(partition)
        return parent_partitions

    def __getattr__(self, name):
        """
        Load the module or the callable
        and fetches an attribute from it.

        Parameters
        ----------
        name: str
            name of the module attribute to fetch

        Returns
        -------
        The fetched attribute from the loaded module or callable
        """
        self.__load_module()

        return getattr(self._mod, name)

    def __getstate__(self):
        return {
            "lib_name": self.lib_name,
            "_mod": None,
            "callable_name": self.callable_name,
        }

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce__(self):
        return (self.__class__, (self.lib_name, self.callable_name))

    def __call__(self, *args, **kwargs):
        """
        Call the callable and returns its output
        if a callable is given as argument.

        Parameters
        ----------
        args: List
            Arguments passed to the callable
        kwargs: Dict
            Optinal arguments passed to the callable

        Raises
        ------
        GuardianAIRuntimeError
            when the callable name is not specified.

        Returns
        -------
        Callable result

        """
        self.__load_module()
        if self.callable_name is None:
            raise GuardianAIRuntimeError(
                "Cannot call a lazy loader when no callable is specified."
            )
        return self._mod(*args, **kwargs)

    @classmethod
    def check_if_partitions_are_installed(cls, partition_names: List[str]) -> bool:
        """Check if specified partitions have been installed.

        Returns True if all packages in the partitions are present in the environment.

        Parameters
        ----------
        partition_names : List[str]
            Names of the partition to be checked.

        Returns
        -------
        bool
            Whether the partition has been installed.
        """
        _get_partitions()
        global __PARTITIONS__
        __PARTITIONS__ = cast(Dict[str, List[str]], __PARTITIONS__)
        installed_pkgs = [p.project_name.lower() for p in pkg_resources.working_set]
        partition_packages: List[str] = []
        for name in partition_names:
            partition_packages += __PARTITIONS__[name]
        for pkg in partition_packages:
            if pkg.lower() not in installed_pkgs:
                return False
        return True

    @classmethod
    def check_if_package_is_installed(cls, package_name: str) -> bool:
        """Return True if specified package has been installed.

        Parameters
        ----------
        package_name : str
            Name of the package to be checked.

        Returns
        -------
        bool
            Whether the package has been installed.
        """
        installed_pkgs = [p.project_name for p in pkg_resources.working_set]
        return package_name in installed_pkgs
