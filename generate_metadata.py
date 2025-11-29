#!/usr/bin/env python3
import argparse
import math
import yaml
import rosbag2_py
from datetime import timedelta, datetime


def to_nanoseconds_duration(duration):
    """
    Convert rosbag2_py BagMetadata.duration or FileInfo.duration to int nanoseconds.
    Handles both datetime.timedelta and plain int.
    """
    if isinstance(duration, int):
        return int(duration)

    if isinstance(duration, timedelta):
        return int(
            duration.days * 24 * 60 * 60 * 1e9
            + duration.seconds * 1e9
            + duration.microseconds * 1e3
        )

    # Some distros expose .nanoseconds
    if hasattr(duration, "nanoseconds"):
        return int(duration.nanoseconds)

    raise TypeError(f"Unsupported duration type: {type(duration)}")


def to_nanoseconds_time(t):
    """
    Convert rosbag2_py BagMetadata.starting_time or FileInfo.starting_time to int nanoseconds since epoch.
    Handles datetime and int.
    """
    if isinstance(t, int):
        return int(t)

    if isinstance(t, datetime):
        return int(t.timestamp() * 1e9)

    # Some distros expose .nanoseconds
    if hasattr(t, "nanoseconds"):
        return int(t.nanoseconds)

    raise TypeError(f"Unsupported time type: {type(t)}")


def bagmetadata_to_yaml_dict(metadata):
    """
    Map rosbag2_py.BagMetadata into a dict matching rosbag2 metadata.yaml format.
    """
    duration_ns = to_nanoseconds_duration(metadata.duration)
    start_ns = to_nanoseconds_time(metadata.starting_time)

    info = {
        "rosbag2_bagfile_information": {
            "version": metadata.version,
            "storage_identifier": metadata.storage_identifier,
            "duration": {"nanoseconds": int(duration_ns)},
            "starting_time": {"nanoseconds_since_epoch": int(start_ns)},
            "message_count": int(metadata.message_count),
            "topics_with_message_count": [],
            "compression_format": metadata.compression_format or "",
            "compression_mode": metadata.compression_mode or "",
            "relative_file_paths": list(metadata.relative_file_paths),
            "files": [],
        }
    }

    # topics_with_message_count
    for t in metadata.topics_with_message_count:
        tm = t.topic_metadata
        info["rosbag2_bagfile_information"]["topics_with_message_count"].append(
            {
                "topic_metadata": {
                    "name": tm.name,
                    "type": tm.type,
                    "serialization_format": tm.serialization_format,
                    # offered_qos_profiles is a single YAML-ish string
                    "offered_qos_profiles": tm.offered_qos_profiles or "",
                },
                "message_count": int(t.message_count),
            }
        )

    # files
    for f in metadata.files:
        file_start_ns = to_nanoseconds_time(f.starting_time)
        file_duration_ns = to_nanoseconds_duration(f.duration)
        info["rosbag2_bagfile_information"]["files"].append(
            {
                "path": f.path,
                "starting_time": {
                    "nanoseconds_since_epoch": int(file_start_ns),
                },
                "duration": {
                    "nanoseconds": int(file_duration_ns),
                },
                "message_count": int(f.message_count),
            }
        )

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Generate ROS 2 style metadata.yaml from an MCAP rosbag"
    )
    parser.add_argument(
        "bag_path",
        help="Path to rosbag (folder or file) – same as you pass to `ros2 bag info`",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="metadata.yaml",
        help="Output metadata YAML path (default: metadata.yaml)",
    )
    parser.add_argument(
        "--storage-id",
        default="mcap",
        help="Storage identifier (default: mcap)",
    )

    args = parser.parse_args()

    # Read metadata using rosbag2_py.Info
    info_reader = rosbag2_py.Info()
    metadata = info_reader.read_metadata(args.bag_path, args.storage_id)

    yaml_dict = bagmetadata_to_yaml_dict(metadata)

    with open(args.output, "w") as f:
        yaml.safe_dump(
            yaml_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    print(f"Wrote metadata to {args.output}")


if __name__ == "__main__":
    main()
