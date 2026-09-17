from pathlib import Path

import pandas as pd

def load_f1_data():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    constructors = pd.read_csv(data_dir / "constructors.csv")
    constructor_standings = pd.read_csv(data_dir / "constructor_standings.csv")
    races = pd.read_csv(data_dir / "races.csv")
    results = pd.read_csv(data_dir / "results.csv")
    return constructors, constructor_standings, races, results
