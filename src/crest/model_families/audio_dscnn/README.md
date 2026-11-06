<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Audio DS-CNN family

This package implements the depthwise-separable CNN family for cached log-mel
classification inputs. The registered family class is `AudioDSCNNFamily`, available
through `from crest.model_families.audio_dscnn import AudioDSCNNFamily`.

[`family.py`](family.py) owns hyperparameter validation, Optuna sampling, the raw
search descriptor and semantic hints, Keras construction, FLOP support, and export
materialization. The architecture combines an initial convolution with repeated
depthwise/pointwise blocks, selectable pooling, and a classification head.
[`decode.py`](decode.py) is the authoritative, ML-runtime-independent source for
categorical choices and stride/channel decoding. Package exports load these modules
lazily, so decoding helpers can be used without importing TensorFlow:

```python
from crest.model_families.audio_dscnn import resolve_stride_schedule
print(resolve_stride_schedule("balanced", 3))
```

Model construction needs TensorFlow and a `ModelBuildContext` supplied by the
pipeline. Dataset preparation and labels belong to the
[dataset adapters](../../datasets/README.md); training loss and metrics belong to the
[sound classification task](../../tasks/README.md). Start experiments from the
[config guide](../../../config/README.md).

See the [model-family contract](../README.md) before changing architecture or export
behavior. Keep decoding and built graphs consistent and extend
[`test_audio_dscnn.py`](../../../../test/test_audio_dscnn.py). Construction and decoding
checks need no board; deployment validation requires the selected device backend.
