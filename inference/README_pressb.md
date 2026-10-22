# PressB inference transport extension

`launch_server.py` uses `packages/openpi_client/websocket_policy_server.py`.
This server retains its existing WebSocket protocol: the first binary frame is
the supplied policy metadata, followed by observation/action exchanges encoded
with the bundled `openpi_client.msgpack_numpy` codec. The different server under
`wall_x/serving/` is not used by this launcher.

Existing clients may continue to send a plain observation dictionary or the RTC
wrapper `{"obs": obs, "is_rtc": bool, "prev_action": optional_array}`. These
requests still pass through `data_processor.in_process` and `policy.infer`, and
receive the original action result and `server_timing`. No new constructor
arguments are required. The launcher also honors its existing `--host` option
and applies an explicit `--model-config.norm-stats-path` override consistently
to both observation preprocessing and model construction; see the
[PiPER deployment guide](README_rtx5090.md).

## Optional split-inference RPC

A host application can supply `rpc_handler` as an additional server constructor
argument. This opts into a separate request envelope:

```python
{
    "__pressb_rpc__": {
        "version": 1,
        "method": "encode",  # encode, decode, release, or health
        "payload": {...},
    }
}
```

The server validates the envelope and calls `rpc_handler` with the inner
dictionary, synchronously on the server event-loop thread. The handler must
return a dictionary. Extension requests bypass the legacy data processor and
policy inference path; the host application supplies the implementation and
any model-specific normalization. This transport hook alone does not add
split encode/decode capability to `WallXPolicy`.

Use the bundled codec for arrays in both requests and responses. The third-party
`msgpack_numpy` package uses a different array representation. Ordinary legacy
requests remain supported even when the extension handler is configured.
Metadata is still exactly the metadata supplied by the host application; the
server does not add capability fields automatically.

Extension failures return a binary structured response:

```python
{
    "protocol_version": 1,
    "error": {"type": "ValueError", "message": "..."},
}
```

`ValueError`, `KeyError`, and `TypeError` leave the connection open so the client
can correct its request. Other exceptions close the connection with code 1011
after sending the error. Legacy request errors retain the original traceback
text frame and connection-close behavior.

## Optional authentication

Supplying `api_key` enables upgrade authentication with the existing client
header format `Authorization: Api-Key <token>`. Omitting it retains unauthenticated
operation. GET `/healthz` remains available without authentication and returns
`OK\n`; it does not expose model state. Model RPC requests use the authenticated
WebSocket connection.
