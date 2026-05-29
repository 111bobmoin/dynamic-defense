# Dynamic Defense CENI Sidecar Bridge

This extension adapts output from a sibling `dynamic_defense_ceni` project into
the local platform input format without modifying dashboard or platform files.

The bridge uses only the Python standard library. It does not import or require
`torch`.

## Files

- `action_logger.py`: append-only UTF-8 JSONL action logger.
- `bridge.py`: reads CENI reports or emits a static P4 validation demo payload.
- `run_bridge.sh`: activates the current repository `.venv` and runs the bridge.
- `validate_bridge_output.py`: validates generated bridge output.

## Run

From the repository root:

```bash
source .venv/bin/activate
bash extensions/dynamic_defense_ceni/run_bridge.sh
```

With no arguments, `run_bridge.sh` runs:

```bash
python extensions/dynamic_defense_ceni/bridge.py --static-demo --once
```

Arguments are passed through to `bridge.py` when provided:

```bash
bash extensions/dynamic_defense_ceni/run_bridge.sh \
  --ceni-project-root /path/to/dynamic_defense_ceni \
  --output-dir /tmp/optimize_multi_vm_runtime/defense_inputs \
  --output-file dynamic_defense.json \
  --once
```

## Inputs

When `--static-demo` is not used, the bridge reads these files from
`--ceni-project-root`:

- `reports/dynamic_defense_summary.json`
- `reports/dynamic_defense_events.csv`
- `runtime/controller_state.json`
- `reports/controller_execution_plan.jsonl`

If those files are not present, use `--static-demo` to generate the P4 final
validation payload.

## Output

The bridge writes JSON atomically by writing a `.tmp` file in the target
directory and then replacing the target file.

Default output:

```text
/tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json
```

The payload contains:

- `status`
- `severity`
- `updated_at`
- `summary`
- `message`
- `metrics`
- `alerts`
- `risk_score`
- `scenario`
- `event_type`
- `affected_links`
- `affected_nodes`
- `recommendation`
- `actions`
- `version`
- `source`

## Validate

```bash
source .venv/bin/activate
python extensions/dynamic_defense_ceni/validate_bridge_output.py \
  --input /tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json
```

The validator prints `PASS` or `FAIL` and appends validation activity to the
same JSONL action log by default:

```text
logs/dynamic_defense_ceni_actions.jsonl
```
