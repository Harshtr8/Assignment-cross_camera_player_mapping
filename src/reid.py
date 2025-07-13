import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_avg_features(data, frame_limit=30):
    """
    Extracts average embeddings for each player ID.
    """
    features = {}
    for d in data:
        if "id" not in d or "embedding" not in d or d["embedding"] is None:
            continue
        pid = d["id"]
        features.setdefault(pid, []).append(np.array(d["embedding"]))

    # Average the first `frame_limit` embeddings for stability
    avg_features = {}
    for pid, feats in features.items():
        if feats:
            avg_features[pid] = np.mean(feats[:frame_limit], axis=0)

    return avg_features

def apply_id_mapping(data_path, id_map, output_path):
    """
    Rewrites the detection file with updated (mapped) IDs.
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    for d in data:
        original_id = d.get("id")
        if original_id in id_map:
            d["id"] = id_map[original_id]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def run_reid(broadcast_json, tacticam_json, output_json):
    """
    Matches players from tacticam to broadcast using average ResNet features.
    """
    print("[INFO] Loading tracked player data...")
    with open(broadcast_json, "r") as f1, open(tacticam_json, "r") as f2:
        broadcast_data = json.load(f1)
        tacticam_data = json.load(f2)

    print("[INFO] Extracting average embeddings...")
    broadcast_feats = extract_avg_features(broadcast_data)
    tacticam_feats = extract_avg_features(tacticam_data)

    if not broadcast_feats:
        print("[ERROR] No valid embeddings found in broadcast_tracked.json.")
        return

    if not tacticam_feats:
        print("[ERROR] No valid embeddings found in tacticam_tracked.json.")
        return

    ids1, vecs1 = zip(*broadcast_feats.items())
    ids2, vecs2 = zip(*tacticam_feats.items())

    print("[INFO] Computing cosine similarities...")
    sim_matrix = cosine_similarity(vecs2, vecs1)

    id_map = {}
    for i, tid in enumerate(ids2):
        best_match_idx = np.argmax(sim_matrix[i])
        best_bid = ids1[best_match_idx]
        id_map[tid] = best_bid

    print("[INFO] Applying broadcast IDs to tacticam...")
    apply_id_mapping(tacticam_json, id_map, output_json)

    print(f"[✅] Cross-camera re-ID complete. Updated file saved to: {output_json}")