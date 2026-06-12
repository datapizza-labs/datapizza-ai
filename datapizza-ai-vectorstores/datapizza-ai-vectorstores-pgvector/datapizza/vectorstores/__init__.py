# Namespace package support for datapizza.vectorstores

from __future__ import annotations

# This file exists so that `datapizza.vectorstores` can be imported both from
# the core datapizza package and from separately-installed vectorstore plugins.
#
# It uses pkgutil.extend_path to support namespace package behavior across
# multiple distributions.

__path__ = __import__("pkgutil").extend_path(__path__, __name__)
