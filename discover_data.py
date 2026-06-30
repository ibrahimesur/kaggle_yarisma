"""
Veri setindeki dosya isimlerini ve sütunları keşfeder.
Kaggle API token ayarlandıktan sonra çalıştırın:
  python discover_data.py
"""
import kagglehub
from pathlib import Path
import pandas as pd

path = kagglehub.competition_download("trendyol-e-ticaret-yarismasi-2026-kaggle")
data_dir = Path(path)

print(f"\nVeri dizini: {data_dir}\n")
print("=" * 60)

for f in sorted(data_dir.iterdir()):
    if f.suffix == ".csv":
        df = pd.read_csv(f, nrows=3)
        print(f"\n📄 {f.name}")
        print(f"   Sütunlar : {list(df.columns)}")
        print(f"   Satır(ör): {len(df)} (ilk 3)")
        print(df.head(3).to_string(index=False))
        print("-" * 60)
    else:
        print(f"\n📁 {f.name} (csv değil)")
