# Third-Party Notices

This file lists third-party source shipped in the artifact. It does not license AEGIS itself.

## DAGER

This package includes a trimmed copy of utilities from *DAGER: Exact Gradient Inversion for Large Language Models* (Petrov, Dimitrov, Baader, Müller, and Vechev; NeurIPS 2024), released under Apache License 2.0 at <https://github.com/insait-institute/dager-gradient-inversion>. The full license is in [`LICENSES/dager/LICENSE.md`](LICENSES/dager/LICENSE.md).

The adapted files live under `src/aegis/third_party/dager/` (`filtering_decoder.py`, `filtering_encoder.py`, `functional.py`, and `partial_models.py`). They were changed for package imports, typing, and the pinned Transformers stack; unused upstream code was dropped.

## Python packages

Dependencies are installed from PyPI at setup time. They are not vendored here and keep their own licenses.
