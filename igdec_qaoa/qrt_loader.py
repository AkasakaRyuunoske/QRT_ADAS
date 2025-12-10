import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


def extract_collisions(scenarios):
    """
    Estrae un flag binario (1 se c'è una collisione, 0 altrimenti) per ogni scenario.
    Controlla 'event_type' direttamente nell'oggetto scenario (che è un singolo evento).
    """
    collision_flags = []

    for s in scenarios:
        has_collision = 1 if s.get("event_type") == "collision" else 0
        collision_flags.append(has_collision)

    return collision_flags


def load_scenarios_from_folder(folder_path):
    """
    Carica tutti i file JSON da una cartella specificata, inclusi quelli nelle sottocartelle.
    Ogni file JSON è atteso essere una LISTA di eventi.
    Il codice estrarrà il PRIMO evento come rappresentativo dello scenario.
    """
    scenarios = []
    print(f"Caricamento scenari dalla cartella: {folder_path} (e sottocartelle)")
    if not os.path.exists(folder_path):
        print(f"⚠️ Attenzione: La cartella '{folder_path}' non esiste.")
        return []

    # Utilizza os.walk per attraversare l'albero delle directory
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)  # Usa 'root' per costruire il percorso completo
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                        if isinstance(data, list) and data:
                            if data:  # Assicurati che la lista non sia vuota
                                scenario_event_data = data[0]
                                scenario_event_data['original_filename'] = filename
                                scenarios.append(scenario_event_data)
                            else:
                                print(f"⚠️ Attenzione: Il file {filename} è una lista vuota. Saltato.")
                        else:
                            print(
                                f"❌ Errore: Il file {filename} non contiene una lista valida di eventi o ha un formato inatteso. Saltato.")

                except json.JSONDecodeError as e:
                    print(f"❌ Errore di decodifica JSON nel file {filename}: {e}")
                except Exception as e:
                    print(f"❌ Errore generico durante la lettura/elaborazione del file {filename}: {e}")
    print(f"Caricati {len(scenarios)} scenari.")
    return scenarios, len(scenarios)


def compute_div_scores(scenarios):
    """
    Calcola il punteggio di diversità (div_score) per ogni scenario.
    Gestisce campi presenti o assenti in base al tipo di evento.
    """
    records = []

    for idx, s in enumerate(scenarios):
        record = {
            "id": idx,
            "town": s.get("town", None),
            "road_type_at_collision": s.get("road_type_at_collision", None)
        }

        record.update(s.get("weather", {}))
        record.update(s.get("town_characteristics", {}))

        records.append(record)

    df = pd.DataFrame(records)

    cat_cols = ['town']
    if 'road_type_at_collision' in df.columns:
        cat_cols.append('road_type_at_collision')

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded_df = pd.DataFrame()
    if not df.empty and any(col in df.columns for col in cat_cols):
        actual_cat_cols = [col for col in cat_cols if col in df.columns]
        if actual_cat_cols:
            cat_encoded = encoder.fit_transform(df[actual_cat_cols])
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(actual_cat_cols))

    all_cols_except_id_cat = [col for col in df.columns if col not in ["id"] + cat_cols]

    numeric_df = df[all_cols_except_id_cat].apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.dropna(axis=1, how='all')
    num_cols_final = numeric_df.columns.tolist()

    scaler = MinMaxScaler()
    num_scaled_df = pd.DataFrame()
    if not numeric_df.empty:
        num_scaled = scaler.fit_transform(numeric_df.fillna(0))
        num_scaled_df = pd.DataFrame(num_scaled, columns=num_cols_final)

    X = pd.concat([num_scaled_df.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)

    if X.empty or X.shape[0] < 2:
        print(
            "Avviso: Meno di 2 scenari o dati insufficienti per calcolare la diversità. Restituendo punteggi di diversità 0.")
        return [0.0] * len(scenarios)

    dist_matrix = pairwise_distances(X, metric='manhattan')

    div_scores = [
        float(np.mean(np.delete(dist_matrix[i], i)))
        for i in range(len(dist_matrix))
    ]

    return div_scores




def load_qrt_df():
    scenarios, tot_test_cases = load_scenarios_from_folder("./../simulation_output")
    collisions = extract_collisions(scenarios)
    div_scores = compute_div_scores(scenarios)

    test_cases_costs = [1] * tot_test_cases


    df = pd.DataFrame({
        "collisions": collisions,
        "div_scores": div_scores,
        "test_cases_costs": test_cases_costs
    })

    scaler = MinMaxScaler()
    df["div_scores_norm"] = scaler.fit_transform(df[["div_scores"]])
    df["div_scores_norm"] = df["div_scores_norm"].round(10)
    del df["div_scores"]

    return df


if __name__ == "__main__":
    print(load_qrt_df())
