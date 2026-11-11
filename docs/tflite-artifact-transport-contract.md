# TFLite artifact transport contract

Status: reviewed and settled on 2026-11-11 against baseline
`d7ad8f5f730c02a09bc4dc3b55170e390fe447e7`. This document describes the
additive boundaries owned by `hci/tflite-transport`.

## Current limitation and reuse decision

The HIL client sends one JSON object and the server always prepares a candidate
from model-family hyperparameters. `CandidatePrepareRequest.tflite_model_path`
is an export destination, so it cannot identify an already exported input
artifact without changing its meaning. A timed-out ZeroMQ REQ socket is left in
the REQ/REP state machine and cannot safely send the next request.

The implementation will retain the existing ZeroMQ endpoint, synchronous
REQ/REP exchange, JSON measurement reply, `HILServer.determine_metrics`, device
controller path, error codes, metric records, and backend cleanup hook. Shared
filesystem paths were rejected because the two hosts do not share a guaranteed
filesystem. Sending a checkpoint was rejected because server-side conversion
would not preserve the host-evaluated bytes. Base64 JSON was rejected because
it expands the payload and erases the explicit binary frame boundary. A second
transfer service or queue is unnecessary.

## Wire protocol

### Capability query

A new client sends this single JSON frame before an artifact request:

```json
{"request_type":"capabilities","protocol_version":1}
```

The server handles it before pipeline bootstrap and replies with JSON:

```json
{
  "request_type":"capabilities",
  "protocol_version":1,
  "artifact_transport_versions":[1],
  "backend":{"name":"<configured device>","supports_tflite_artifact":true},
  "limits":{"max_artifact_bytes":16777216}
}
```

The backend support value is resolved through the configured device registry
and options without loading the dataset, task, or model family. It says only
that the selected backend accepts a supplied TFLite file; it does not attest to
board reachability or health. A timeout, malformed response, missing version,
or false backend support is an incompatibility. The client resets its REQ
socket and raises a typed transport error; it never probes support by sending
multipart data and never falls back to an architecture-only request.

Legacy clients continue to send their existing JSON candidate frame. A new
server accepts it unchanged. A new client can use the legacy path without a
capability query when no artifact is supplied. Old servers will return their
ordinary structured request error for the query or time out; either result is
reported as unsupported before multipart transfer.

### Artifact request

Protocol version 1 is exactly two ZeroMQ frames. Frame 0 is UTF-8 JSON and
frame 1 is the raw flatbuffer:

```json
{
  "request_type":"candidate_artifact",
  "protocol_version":1,
  "request_id":"<stable non-empty client identity>",
  "family_hparams":{},
  "runtime_metadata":{},
  "quantization_mode":"int8_ptq",
  "device_options_overrides":null,
  "artifact":{
    "format":"tflite",
    "byte_length":1234,
    "sha256":"<64 lowercase hexadecimal characters>",
    "inputs":[{"shape":[1,4],"dtype":"int8","quantization":{"scales":[0.015625],"zero_points":[-3],"quantized_dimension":0}}],
    "outputs":[{"shape":[1,3],"dtype":"int8","quantization":{"scales":[0.0078125],"zero_points":[-128],"quantized_dimension":0}}]
  }
}
```

`request_id` and `artifact.sha256` form the retry identity and remain identical
on every attempt. `family_hparams` and `runtime_metadata` retain their current
types so metric collection and evidence callers need no parallel schema.
Tensor metadata is declarative evidence, not authority. `inputs` and `outputs`
are ordered, non-empty lists preserving every interpreter tensor, including
supported multi-input or multi-output models. Each entry contains its complete
integer `shape`, normalized NumPy `dtype` name, and a `quantization` object
with ordered finite `scales`, ordered integer `zero_points`, and integer
`quantized_dimension`. Float tensors use empty scale and zero-point lists.
The server compares this declaration exactly with the actual interpreter
metadata and request quantization mode. Missing, reordered, omitted, or
ambiguous scalar quantization metadata is a request error.

The receive boundary rejects any frame count other than one for legacy/query
or two for an artifact request. It rejects invalid JSON, unsupported version or
format, missing identity, non-integer/negative or oversized lengths, length
mismatch, hash mismatch, invalid TFLite, and incompatible tensor metadata. A
sender-supplied path is never accepted. Every rejection produces the existing
JSON-shaped failure reply so the REP socket returns to receive state.

Successful artifact replies retain all current metric fields and add:

```json
{"request_id":"...","artifact_sha256":"...","artifact_byte_length":1234}
```

Failure replies include the identities when they were validated far enough to
be trustworthy. Existing consumers can ignore these additive keys.
The client compares all three fields with the sent logical request before
accepting successful metrics. A missing or different identity field raises
typed transport-integrity exhaustion; tampered success data never reaches
objective scoring.

## Staging and evidence

Validated bytes are written beneath the server-configured staging root in a
server-created unique directory; neither request identity nor any metadata is
used directly as a path component. The input file is opened exclusively and
hashed again from disk before preparation. The server owns this directory and
removes it in `finally` after backend cleanup, on request rejection, and on an
expected preparation failure. Cleanup failures are logged without replacing an
active failure.

Generic request evidence extends the current request record rather than adding
a second database. The record includes request ID, artifact hash and length,
staged path, configured backend, result association, cleanup outcome, and
exception text. Evidence location follows existing output conventions; a new
`network.artifact_staging_dir` controls disposable staging only. Durable
evidence never stores a second copy of the model unless an existing diagnostic
retention setting explicitly requires it.

Legacy evidence behavior remains unchanged. The current STM32 record remains
mandatory with its existing failure policy. Generic artifact evidence is
required only for multipart requests; legacy requests for other boards do not
begin failing because a new evidence directory is absent or unwritable.

## Candidate preparation boundary

`CandidatePrepareRequest` gains one optional field:

```python
source_tflite_path: Path | None = None
```

Absent remains the default and preserves every existing caller and legacy
backend path. When present, the path is the server-owned, already validated
input file. It is distinct from `tflite_model_path`, which remains an export
destination for legacy preparation. The two modes are exclusive in behavior:
the backend must consume/copy the source bytes without converting,
recalibrating, or materializing a model. The orchestration passes `model=None`,
`calibration_split=None`, and `checkpoint_path=None` only after positive
backend support validation.
When `source_tflite_path` is present, a non-`None` model, calibration split,
or checkpoint is a request error. The backend also validates standalone local
calls: the source must be a regular readable file and conflicting inputs are
rejected even when orchestration is bypassed.

`DeviceInterface` gains one default-false capability method:

```python
def supports_tflite_artifact(self) -> bool:
    return False
```

This narrow method is needed because `requires_candidate_model()` means that a
backend needs generated model build inputs; changing it based on a particular
request would break its argument-free capability contract. Registry/options
construction already selects the concrete backend, so a second capability
registry is unnecessary. Backends opt in only when their `prepare_candidate`
implementation supports the source field. Presence plus false support returns
a request/configuration failure before bypassing
`requires_candidate_model()` or `requires_training_data()`.

`DeviceInterface.prepare_candidate` keeps its signature and `Path` return
contract. `crest.devices` adds `CandidatePreparationError`, a narrow shared
`RuntimeError` base carrying stable `error_kind` and detail. Opted-in backends
raise it for expected staging/tool failures and may subclass it without server
imports. The preparation boundary catches `CandidatePreparationError`,
`serial.SerialException`, and `OSError` from staging/tool filesystem work,
then returns structured failure metrics and cleans up. Supplied-model
`ValueError` becomes a structured request error. Tool-specific subprocess
failures must be wrapped in `CandidatePreparationError`; arbitrary exceptions
propagate.

Callers affected are `HILServer._build_candidate_prepare_request`, the server
candidate orchestration, concrete backends that opt in, tests constructing the
dataclass, and Plan 4's client call. Existing positional construction remains
compatible because the optional field is last. No ABC, `FitPlan`, optimizer,
or metric-record signature changes.

## Client API and recovery

The downstream APIs are:

```python
_ensure_hil_artifact_support(*, refresh: bool = False) -> dict
_hil_request(payload, *, artifact_path: Path | None = None) -> dict
```

Plan 4 calls `_ensure_hil_artifact_support(refresh=True)` once per NAS/smoke
invocation, including resumes, before training. It negotiates v1, confirms
backend support, validates the advertised limit, and
caches the response for the endpoint. `_hil_request` uses the same helper
before multipart transmission if no valid cache exists. Socket reset clears
the cache and retry renegotiates. Refresh performs the normal socket reset so a
cached response cannot certify a server replaced between invocations; it does
not add another recovery mechanism.

With no artifact it sends the existing single JSON frame. With an artifact it
reads the file once per logical request, derives SHA-256/length and tensor
metadata, performs capability negotiation, and sends the two-frame request.
Plan 4 passes the already host-evaluated export path to this keyword and does
not build wire metadata or retry logic itself.

`network.max_artifact_bytes` (positive integer),
`network.artifact_staging_dir` (server path), and
`network.hil_request_retries` (non-negative integer, default `1`) are additive.
Existing send/receive timeout keys remain authoritative and all historical
configs use the defaults. Invalid booleans, strings where integers are needed,
zero payload limits, and negative retries fail configuration validation.

On send/receive timeout or recoverable ZMQ state failure, the client closes the
REQ socket with `linger=0`, clears it, recreates it lazily, and retries up to
the configured count. Capability negotiation uses the same reset rule. The
logical `request_id`, bytes, hash, length, and metadata are reused exactly.
Exhaustion raises `HILTransportError`, a
`MeasurementInfrastructureError` subclass with stable constructor metadata:
`endpoint`, `request_id`, `attempts`, `phase` (`capability`, `send`,
`receive`, or `integrity`), and chained cause. Plan 4 catches the existing
base and records it through the failure guard. A valid capability reply lacking
protocol/backend support raises separate `HILArtifactCompatibilityError` with
endpoint, advertised versions, backend, and reason. Local validation raises
`ValueError`. Capability timeout after bounded reset remains transport
exhaustion. Retries are at-least-once and may repeat physical measurement.

## Migration and validation

No config edit is needed for legacy JSON requests. Artifact clients require a
new server and an opted-in backend, and fail before transfer when either is
missing. During integration, Plan 1 implements the capability and source-file
branch; Plan 4 uses only the optional `_hil_request` argument.

Focused tests will cover legacy old-client/new-server JSON behavior;
new-client/old-server query error, malformed reply, and timeout; capability
handling before bootstrap and the Plan 4 pre-training helper seam; exact byte/hash round trip through a tiny valid
TFLite; actual tensor/quantization validation; malformed frame counts, length,
hash, size, format, and model; unique staging and cleanup on success/failure;
unsupported backend rejection before model/data bootstrap; no reconversion;
expected preparation failure followed by a successful request; generic request
evidence; lost reply and delayed reply recovery; identical identity across
retry; tampered successful reply identity/hash/length; a delayed real-ZeroMQ
recovery; successful next request; and typed exhaustion. Plan 4 owns the combined
test that typed exhaustion populates trial infrastructure attributes and the
three-strike guard.
