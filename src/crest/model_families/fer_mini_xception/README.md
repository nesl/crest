<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# FER Mini-Xception family

This package adapts the HCI project's five-class Mini-Xception model to the
CREST model-family contract. The learned topology is derived from Octavio
Arriaga's MIT-licensed Mini-Xception work; the retained license is in
[`THIRD_PARTY_LICENSE.txt`](THIRD_PARTY_LICENSE.txt).

## Audited source evidence

The HCI training directory was not a Git checkout when this adapter was
prepared. The exact reviewed inputs are therefore pinned by SHA-256. Paths are
relative to `HCI-final-project/training`.

| HCI source | SHA-256 | Reused behavior |
|---|---|---|
| `hci_fer/mixed.py` | `2f56fb35cdf40fac70f9131eaa58e4b1fcde00797d89373a0ec5ea3802c43c01` | mixed-source loading, source ordering and equal-source weighting |
| `hci_fer/ferplus.py` | `913b29e767b6b10ff1f8ac1fda577f5a69c26a9f104000d0d62d3ef5e8dec63b` | verified FER+ provenance and official split policy |
| `hci_fer/data.py` | `a558614ff4e3f2a5b976f56d0436c7097cee1379b66aac25cf5927b5b50a8727` | class order and nearest-floor normalization |
| `hci_fer/targets.py` | `82ca7a9559c5a2d07dbc6dcccc37ea0ae26f044c1797d9b18cd7d94eb38ac9cf` | conditional five-class FER+ targets |
| `hci_fer/model.py` | `30016c2990ac37caa3d77a3a274b9fd677c40035abe3c4a281f34c529074b583` | stem, residual separable blocks, SE attention and probability head |
| `hci_fer/pipeline.py` | `ac3dac201883623ea862c4b7ca3654dbb261de7f2b4697e7d9cf8427c6fb5d68` | augmentation, loss, optimizer and evaluation semantics |
| `hci_fer/checkpoints.py` | `3ab44f1dd02240d1bd60c01ef42e3ec74d329196dd36e6d10a636ce1fe5d43e6` | validation-loss checkpoint policy and diagnostic macro-F1 reporting |
| `configs/mini_xception_mixed_se_96.json` | `9eaed7370359ce51966543b3435b13990912ff6adcb90d2061eac5788035c928` | fixed 96-pixel reference settings |
| `data/mixed_ferplus_raf_v1/manifest.json` | `51f807a6bbdbbddddc44a4380873d31ba8961ffd2f1c8d51b8f5b38bc5c9aed1` | prepared-data schema, split identities and source hashes |
| `third_party/MINI_XCEPTION_LICENSE.txt` | `032cdce3f236ea4b7d23a63a00ee9f476d01947fba2443d885599c1ad4e5dece` | upstream MIT notice |

The selected HCI `mixed_ferplus_raf_se_96_v1` model has 61,453 total
parameters, uses widths `16, 32, 64, 128`, applies SE attention to the last two
blocks, and selects its primary checkpoint by validation loss including model
regularization. CREST retains those learned computations behind a logical
`(height, width)` input and an explicit Keras reshape to a singleton channel.

Dataset loading, training orchestration, TFLite conversion, host inference,
and device measurement remain in their existing CREST owners. This package
contains only architecture decoding and model construction.

## Active export-stack evidence

The CREST TensorFlow 2.20/Keras 3.14 environment retains the logical input
adapter as `RESHAPE` and also emits `SHAPE`, `STRIDED_SLICE`, and `PACK` to
construct its dynamic batch shape. A focused full-INT8 smoke artifact also
contains the expected convolution, depthwise convolution, pooling, add, and
softmax operations and runs successfully through CREST's existing host TFLite
predictor. `DELEGATE` is interpreter execution metadata rather than a
flatbuffer operation and is excluded from the recorded operator set.

This is host conversion/interpretation evidence only. It does not claim that
the artifact has compiled or run on a microcontroller. Device backends must
check the actual artifact's operator versions and resolver registrations.
