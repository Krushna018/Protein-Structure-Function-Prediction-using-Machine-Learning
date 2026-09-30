import requests
import json
import pandas as pd
import time
import os

EC_CLASSES = {
    1: "Oxidoreductase",
    2: "Transferase",
    3: "Hydrolase",
    4: "Lyase",
    5: "Isomerase",
    6: "Ligase"
}

def fetch_pdb_ids(ec_class, max_results=800):
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
                        "operator": "exact_match",
                        "value": str(ec_class)
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 2.5
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "greater_or_equal",
                        "value": 50
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "less_or_equal",
                        "value": 800
                    }
                }
            ]
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results
            }
        }
    }
    
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    for attempt in range(5):
        try:
            response = requests.post(url, json=query, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return [x["identifier"] for x in data.get("result_set", [])]
            else:
                time.sleep(2)
        except Exception as e:
            time.sleep(2)
    return []

def fetch_entity_data(entity_ids):
    all_results = {}
    batch_size = 50
    for i in range(0, len(entity_ids), batch_size):
        batch = entity_ids[i:i+batch_size]
        query = """
        query($ids: [String!]!) {
          polymer_entities(entity_ids: $ids) {
            rcsb_id
            entity_poly {
              pdbx_seq_one_letter_code_can
            }
            polymer_entity_instances {
              rcsb_polymer_instance_feature {
                type
                feature_positions {
                  beg_seq_id
                  end_seq_id
                }
              }
            }
          }
        }
        """
        for attempt in range(5):
            try:
                response = requests.post("https://data.rcsb.org/graphql", json={'query': query, 'variables': {'ids': batch}}, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and data["data"]["polymer_entities"]:
                        for entity in data["data"]["polymer_entities"]:
                            if entity:
                                all_results[entity["rcsb_id"]] = entity
                    break
                else:
                    time.sleep(2)
            except Exception as e:
                print(f"Retry {attempt+1} for batch {batch[0]}: {e}")
                time.sleep(5)
        time.sleep(0.1)
    return all_results

def main():
    print("Starting data collection...")
    os.makedirs("data/raw", exist_ok=True)
    
    # Load existing to resume if possible
    records = []
    seen_sequences = set()
    if os.path.exists("data/raw/dataset.csv"):
        existing_df = pd.read_csv("data/raw/dataset.csv")
        records = existing_df.to_dict('records')
        seen_sequences = set(existing_df['sequence'].values)
        print(f"Resuming with {len(records)} existing records.")
    
    # Only fetch missing EC classes
    completed_ecs = set([r['ec_class'] for r in records]) if records else set()
    
    for ec, ec_name in EC_CLASSES.items():
        if ec in completed_ecs and len([r for r in records if r['ec_class'] == ec]) >= 100:
            print(f"Skipping {ec_name} (already collected)")
            continue
            
        print(f"Fetching IDs for {ec_name} (EC {ec})...")
        ids = fetch_pdb_ids(ec, max_results=800)
        print(f"Found {len(ids)} candidates.")
        
        print(f"Fetching sequence and structure features...")
        entity_data = fetch_entity_data(ids)
        
        count = 0
        for pdb_id, data in entity_data.items():
            if not data.get("entity_poly") or not data.get("polymer_entity_instances"):
                continue
            seq = data["entity_poly"].get("pdbx_seq_one_letter_code_can")
            if not seq or "X" in seq:
                continue
            
            if seq in seen_sequences:
                continue
            seen_sequences.add(seq)
            
            instances = data.get("polymer_entity_instances", [])
            helix_frac = 0.0
            sheet_frac = 0.0
            
            if instances:
                features = instances[0].get("rcsb_polymer_instance_feature", [])
                if features:
                    helix_residues = set()
                    sheet_residues = set()
                    for f in features:
                        ftype = f.get("type")
                        if ftype in ["HELIX_P"]:
                            for pos in f.get("feature_positions", []):
                                beg = pos.get("beg_seq_id")
                                end = pos.get("end_seq_id")
                                if beg is not None and end is not None:
                                    helix_residues.update(range(beg, end + 1))
                        elif ftype in ["SHEET"]:
                            for pos in f.get("feature_positions", []):
                                beg = pos.get("beg_seq_id")
                                end = pos.get("end_seq_id")
                                if beg is not None and end is not None:
                                    sheet_residues.update(range(beg, end + 1))
                    
                    helix_frac = len(helix_residues) / len(seq)
                    sheet_frac = len(sheet_residues) / len(seq)
            
            records.append({
                "pdb_id": pdb_id,
                "ec_class": ec,
                "ec_name": ec_name,
                "sequence": seq,
                "length": len(seq),
                "helix_fraction": helix_frac,
                "sheet_fraction": sheet_frac,
                "coil_fraction": 1.0 - (helix_frac + sheet_frac)
            })
            count += 1
            if count >= 600:
                break
                
        print(f"Successfully processed {count} unique sequences for {ec_name}.")
        
        # Save after each EC class
        df = pd.DataFrame(records)
        df.to_csv("data/raw/dataset.csv", index=False)

    df = pd.DataFrame(records)
    df.to_csv("data/raw/dataset.csv", index=False)
    print(f"Data collection complete! Saved {len(df)} records to data/raw/dataset.csv")

if __name__ == "__main__":
    main()
