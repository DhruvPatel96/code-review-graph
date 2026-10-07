"""Storage contracts and backend implementations.

The legacy ``graph.GraphStore`` remains the SQLite implementation while its
storage operations are extracted incrementally. Import contracts from
``storage.base``, data objects from ``storage.models``, and the opener from
``storage.factory``. Keeping this package initializer empty avoids loading a
database implementation just to import a contract or a data object.
"""
