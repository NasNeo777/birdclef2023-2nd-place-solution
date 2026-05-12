import json
import os


def normalize_audio_key(value):
    base = os.path.basename(str(value).strip())
    stem, _ = os.path.splitext(base)
    return stem


def _iter_pseudo_groups(pseudo):
    if not isinstance(pseudo, dict):
        return

    if "pseudo" in pseudo and isinstance(pseudo["pseudo"], list):
        yield "default", pseudo
        return

    for group_name, group in pseudo.items():
        if isinstance(group, dict) and "pseudo" in group and isinstance(group["pseudo"], list):
            yield group_name, group


def load_pseudo_labels(pseudo_file, hand_label_file=None):
    with open(pseudo_file) as f:
        pseudo = json.load(f)

    if hand_label_file and os.path.exists(hand_label_file):
        with open(hand_label_file) as f:
            hand_label = json.load(f)
        merge_hand_labels(pseudo, hand_label)

    return pseudo


def merge_hand_labels(pseudo, hand_label):
    pseudo_entries_by_version = {}
    for _, group in _iter_pseudo_groups(pseudo):
        for oof in group.get("pseudo", []):
            for version, file_preds in oof.get("pred", {}).items():
                normalized_map = {
                    normalize_audio_key(file_key): file_key for file_key in file_preds.keys()
                }
                pseudo_entries_by_version.setdefault(str(version), []).append(
                    (file_preds, normalized_map)
                )

    for version, version_preds in hand_label.get("pred", {}).items():
        entries = pseudo_entries_by_version.get(str(version), [])
        if not entries:
            continue

        for filename, label_preds in version_preds.items():
            normalized_filename = normalize_audio_key(filename)
            for file_preds, normalized_map in entries:
                raw_filename = normalized_map.get(normalized_filename)
                if raw_filename is None:
                    continue

                pseudo_label_preds = file_preds.get(raw_filename, {})
                for label, seconds in label_preds.items():
                    if label not in pseudo_label_preds:
                        continue
                    for second, value in seconds.items():
                        if second in pseudo_label_preds[label]:
                            pseudo_label_preds[label][second] = value
