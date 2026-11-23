# Utilities

Small, reusable helpers independent of dataset, search, and hardware policies.

- `geometry.py`: Quaternion, Euler-angle, and rotation-matrix conversions.
- `windowing.py`: NumPy sliding windows aligned to the final input sample.

Keep domain-specific logic in its owning package.

Use `crest.utils.geometry` in new code; `crest.geometry` remains a compatibility import.
