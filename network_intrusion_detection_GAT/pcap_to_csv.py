from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from src.cicflow_adapter import adapt_cicflowmeter_schema
from src.data import LABEL_COLUMN
from src.output_layout import make_stage_dir, write_manifest


PCAP_SUFFIXES = {".pcap", ".pcapng"}
PROTO253 = 253
INNER_PROTO_OFFSET = 4
ENCAP_HEADER_LEN = 29
SUPPORTED_INNER_PROTOCOLS = {1, 6, 17}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PCAP/PCAPNG traffic into CICIDS-style CSV with cicflowmeter."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="A .pcap/.pcapng file or a directory that contains them.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output CSV path for a single file or merged run, or an output directory for batch mode.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Dedicated result directory that stores converted CSVs under <result-dir>/pcap_csv/.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label injected as the 'Label' column. Leave unset for unlabeled inference input.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="When --input-path is a directory, merge all converted flows into one CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose cicflowmeter logging.",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Keep node attribution columns such as src_ip, dst_ip, src_port, protocol, and timestamp.",
    )
    return parser.parse_args()


def discover_pcap_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in PCAP_SUFFIXES:
            raise ValueError(f"Unsupported file type: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            [
                *input_path.glob("*.pcap"),
                *input_path.glob("*.pcapng"),
            ]
        )
        if not files:
            raise FileNotFoundError(f"No .pcap or .pcapng files found in {input_path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def validate_output_args(output_path: Path | None, result_dir: Path | None) -> None:
    if output_path is not None and result_dir is not None:
        raise ValueError("Use either --output-path or --result-dir, not both.")


def make_default_output_root(input_path: Path, result_dir: Path | None) -> Path:
    return make_stage_dir(input_path=input_path, stage="pcap_csv", result_dir=result_dir)


def make_workspace_temp_root() -> Path:
    temp_root = Path("outputs") / "_internal" / "cicflowmeter_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    return temp_root


def decapsulate_proto253_packet(packet: Any) -> tuple[Any | None, dict[str, Any] | None]:
    try:
        from scapy.all import Ether, IP
    except ImportError as exc:
        raise RuntimeError(
            "scapy is required for proto-253 decapsulation. Run `pip install -r requirements.txt` first."
        ) from exc

    if IP not in packet:
        return packet, None

    ip_layer = packet[IP]
    if int(ip_layer.proto) != PROTO253:
        return packet, None

    payload = bytes(ip_layer.payload)
    if len(payload) < ENCAP_HEADER_LEN:
        return None, {"status": "dropped_short_payload"}

    inner_proto = int(payload[INNER_PROTO_OFFSET])
    if inner_proto not in SUPPORTED_INNER_PROTOCOLS:
        return None, {"status": "dropped_unsupported_inner_proto", "inner_proto": inner_proto}

    inner_payload = payload[ENCAP_HEADER_LEN:]
    rebuilt_ip = ip_layer.copy()
    rebuilt_ip.remove_payload()
    rebuilt_ip.proto = inner_proto
    if hasattr(rebuilt_ip, "len"):
        del rebuilt_ip.len
    if hasattr(rebuilt_ip, "chksum"):
        del rebuilt_ip.chksum

    if Ether in packet:
        eth_layer = packet[Ether].copy()
        eth_layer.remove_payload()
        rebuilt_packet = Ether(bytes(eth_layer / (rebuilt_ip / inner_payload)))
    else:
        rebuilt_packet = IP(bytes(rebuilt_ip / inner_payload))

    if hasattr(packet, "time"):
        rebuilt_packet.time = packet.time
    return rebuilt_packet, {"status": "decapsulated", "inner_proto": inner_proto}


def resolve_single_output(input_file: Path, output_path: Path | None, default_root: Path) -> Path:
    if output_path is None:
        return default_root / f"{input_file.stem}_flows.csv"
    if output_path.suffix.lower() == ".csv":
        return output_path
    return output_path / f"{input_file.stem}_flows.csv"


def resolve_batch_output_dir(output_path: Path | None, default_root: Path) -> Path:
    if output_path is None:
        return default_root
    if output_path.suffix.lower() == ".csv":
        raise ValueError("Batch mode without --merge requires --output-path to be a directory.")
    return output_path


def resolve_merged_output(output_path: Path | None, default_root: Path) -> Path:
    if output_path is None:
        return default_root / "merged_flows.csv"
    if output_path.suffix.lower() == ".csv":
        return output_path
    return output_path / "merged_flows.csv"


def run_cicflowmeter(input_file: Path, raw_output_csv: Path, verbose: bool) -> dict[str, Any]:
    try:
        from cicflowmeter.flow_session import FlowSession
        from scapy.utils import PcapReader
    except ImportError as exc:
        raise RuntimeError(
            "cicflowmeter is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    raw_output_csv.parent.mkdir(parents=True, exist_ok=True)
    session = FlowSession(
        output_mode="csv",
        output=str(raw_output_csv),
        fields=None,
        verbose=verbose,
    )
    stats: dict[str, Any] = {
        "total_packets": 0,
        "packets_seen_by_cicflowmeter": 0,
        "outer_proto253_packets": 0,
        "decapsulated_packets": 0,
        "dropped_short_payload_packets": 0,
        "dropped_unsupported_inner_proto_packets": 0,
        "passthrough_packets": 0,
        "ignored_non_flow_ip_packets": 0,
        "ignored_non_ip_packets": 0,
        "inner_protocol_counts": {"1": 0, "6": 0, "17": 0},
    }

    reader = None
    try:
        reader = PcapReader(str(input_file))
        for packet in reader:
            stats["total_packets"] += 1
            processed_packet, decap_info = decapsulate_proto253_packet(packet)

            if decap_info is not None:
                stats["outer_proto253_packets"] += 1
                status = str(decap_info["status"])
                if status == "decapsulated":
                    stats["decapsulated_packets"] += 1
                    inner_proto = str(int(decap_info["inner_proto"]))
                    stats["inner_protocol_counts"][inner_proto] = stats["inner_protocol_counts"].get(inner_proto, 0) + 1
                elif status == "dropped_short_payload":
                    stats["dropped_short_payload_packets"] += 1
                    continue
                elif status == "dropped_unsupported_inner_proto":
                    stats["dropped_unsupported_inner_proto_packets"] += 1
                    continue

            else:
                processed_packet = packet
                if "IP" in packet:
                    stats["passthrough_packets"] += 1

            if "IP" not in processed_packet:
                stats["ignored_non_ip_packets"] += 1
                continue
            if "TCP" in processed_packet or "UDP" in processed_packet:
                session.process(processed_packet)
                stats["packets_seen_by_cicflowmeter"] += 1
                continue
            stats["ignored_non_flow_ip_packets"] += 1
    finally:
        if reader is not None:
            reader.close()
        session.flush_flows()
    return stats


def load_raw_flow_dataframe(raw_output_csv: Path) -> pd.DataFrame:
    if not raw_output_csv.exists() or raw_output_csv.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(raw_output_csv, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def normalize_flow_dataframe(df: pd.DataFrame, label: str | None, keep_metadata: bool) -> pd.DataFrame:
    normalized = adapt_cicflowmeter_schema(
        df,
        label_column=LABEL_COLUMN,
        drop_metadata=not keep_metadata,
        drop_extra=True,
        ensure_all_features=True,
    )
    if label is not None:
        normalized[LABEL_COLUMN] = str(label).strip()
    return normalized


def write_frame(df: pd.DataFrame, output_csv: Path, *, append: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_csv,
        index=False,
        mode="a" if append else "w",
        header=not append,
        encoding="utf-8" if append else "utf-8-sig",
    )


def cleanup_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _convert_one_file_impl(
    input_file: Path,
    output_csv: Path,
    label: str | None,
    verbose: bool,
    keep_metadata: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    temp_root = make_workspace_temp_root()
    raw_output_csv = temp_root / f"{input_file.stem}_raw.csv"
    try:
        cicflow_stats = run_cicflowmeter(input_file=input_file, raw_output_csv=raw_output_csv, verbose=verbose)
        frame = normalize_flow_dataframe(
            load_raw_flow_dataframe(raw_output_csv),
            label=label,
            keep_metadata=keep_metadata,
        )
    finally:
        cleanup_file(raw_output_csv)
    write_frame(frame, output_csv, append=False)
    return frame, {"proto253_decapsulation": cicflow_stats}


def convert_one_file(
    input_file: Path,
    output_csv: Path,
    label: str | None,
    verbose: bool,
    keep_metadata: bool,
    *,
    return_metadata: bool = False,
) -> int | tuple[int, dict[str, Any]]:
    frame, metadata = _convert_one_file_impl(
        input_file=input_file,
        output_csv=output_csv,
        label=label,
        verbose=verbose,
        keep_metadata=keep_metadata,
    )
    if return_metadata:
        return len(frame), metadata
    return len(frame)


def convert_many_files(
    input_files: list[Path],
    output_path: Path,
    *,
    label: str | None,
    merge: bool,
    verbose: bool,
    keep_metadata: bool,
) -> int | tuple[int, list[dict[str, Any]]]:
    total_rows = 0
    file_metadata: list[dict[str, Any]] = []
    temp_root = make_workspace_temp_root()
    for index, input_file in enumerate(input_files):
        single_output = (
            temp_root / f"{index:04d}_{input_file.stem}_merge.csv"
            if merge
            else output_path / f"{input_file.stem}_flows.csv"
        )
        frame, metadata = _convert_one_file_impl(
            input_file=input_file,
            output_csv=single_output,
            label=label,
            verbose=verbose,
            keep_metadata=keep_metadata,
        )
        rows = len(frame)
        total_rows += rows
        file_metadata.append(
            {
                "input_pcap": str(input_file),
                "output_csv": str(output_path if merge else single_output),
                "rows_written": rows,
                "keep_metadata": bool(keep_metadata),
                "label": label,
                **metadata,
            }
        )
        if merge:
            write_frame(frame, output_path, append=index > 0)
            cleanup_file(single_output)
    return total_rows, file_metadata


def main() -> None:
    args = parse_args()
    validate_output_args(args.output_path, args.result_dir)
    input_files = discover_pcap_files(args.input_path)
    default_root = (
        make_default_output_root(args.input_path, args.result_dir)
        if args.output_path is None
        else Path(".")
    )
    manifest_entries: list[dict[str, Any]] = []

    if args.input_path.is_file():
        output_csv = resolve_single_output(args.input_path, args.output_path, default_root)
        rows, metadata = convert_one_file(
            input_file=args.input_path,
            output_csv=output_csv,
            label=args.label,
            verbose=args.verbose,
            keep_metadata=args.keep_metadata,
            return_metadata=True,
        )
        manifest_entries.append(
            {
                "input_pcap": str(args.input_path),
                "output_csv": str(output_csv),
                "rows_written": rows,
                "keep_metadata": bool(args.keep_metadata),
                "label": args.label,
                **metadata,
            }
        )
        if args.output_path is None:
            write_manifest(
                default_root.parent / "pcap_to_csv_manifest.json",
                {
                    "script": "pcap_to_csv.py",
                    "input_path": str(args.input_path),
                    "output_root": str(default_root.parent),
                    "files": manifest_entries,
                },
            )
        print(f"Input PCAP: {args.input_path}")
        print(f"Output CSV: {output_csv}")
        print(f"Rows written: {rows}")
        return

    if args.merge:
        output_csv = resolve_merged_output(args.output_path, default_root)
        rows, file_metadata = convert_many_files(
            input_files=input_files,
            output_path=output_csv,
            label=args.label,
            merge=True,
            verbose=args.verbose,
            keep_metadata=args.keep_metadata,
        )
        manifest_entries.extend(file_metadata)
        if args.output_path is None:
            write_manifest(
                default_root.parent / "pcap_to_csv_manifest.json",
                {
                    "script": "pcap_to_csv.py",
                    "input_path": str(args.input_path),
                    "output_root": str(default_root.parent),
                    "files_processed": len(input_files),
                    "rows_written": rows,
                    "merge": True,
                    "files": manifest_entries,
                },
            )
        print(f"Input directory: {args.input_path}")
        print(f"Merged output CSV: {output_csv}")
        print(f"Files processed: {len(input_files)}")
        print(f"Rows written: {rows}")
        return

    output_dir = resolve_batch_output_dir(args.output_path, default_root)
    rows, file_metadata = convert_many_files(
        input_files=input_files,
        output_path=output_dir,
        label=args.label,
        merge=False,
        verbose=args.verbose,
        keep_metadata=args.keep_metadata,
    )
    manifest_entries.extend(file_metadata)
    if args.output_path is None:
        write_manifest(
            default_root.parent / "pcap_to_csv_manifest.json",
            {
                "script": "pcap_to_csv.py",
                "input_path": str(args.input_path),
                "output_root": str(default_root.parent),
                "files_processed": len(input_files),
                "rows_written": rows,
                "files": manifest_entries,
            },
        )
    print(f"Input directory: {args.input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Files processed: {len(input_files)}")
    print(f"Rows written: {rows}")


if __name__ == "__main__":
    main()
