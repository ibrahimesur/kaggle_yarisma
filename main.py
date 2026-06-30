"""
============================================================================
  Trendyol E-Ticaret Yarışması 2026 – Kaggle Datathon
  Baseline CatBoost Pipeline (main.py)
  ───────────────────────────────────────────────────────
  Adımlar:
    1. Veri okuma (kagglehub ile dinamik path)
    2. Negatif örnekleme (%50 pozitif / %50 negatif)
    3. Özellik mühendisliği & metin ön işleme
    4. Stratified 5-Fold CatBoost eğitimi + Macro-F1
    5. Tahmin & submission.csv üretimi
    6. Kaggle CLI ile otomatik gönderim
============================================================================
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════════════════
# 1. VERİ OKUMA
# ═══════════════════════════════════════════════════════════════════════════

def _find_csv(data_dir: Path, candidates: list, description: str) -> Path:
    """Verilen aday isimlerden ilk bulunanı döndürür, bulunamazsa hata fırlatır."""
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"[HATA] '{description}' dosyası bulunamadı.\n"
        f"  Aranan isimler: {candidates}\n"
        f"  Dizin: {data_dir}\n"
        f"  Mevcut dosyalar: {[f.name for f in data_dir.iterdir()]}"
    )


def load_data() -> tuple:
    """kagglehub üzerinden veri setini indirir ve DataFrame olarak döndürür."""
    import kagglehub
    path = kagglehub.competition_download("trendyol-e-ticaret-yarismasi-2026-kaggle")
    data_dir = Path(path)
    print(f"[INFO] Veri dizini: {data_dir}")

    all_files = [f.name for f in data_dir.iterdir()]
    print(f"[INFO] Dizindeki dosyalar: {all_files}")

    # Dosya isimlerini otomatik keşfet (olası alternatif isimler dahil)
    items_path = _find_csv(data_dir,
        ["items.csv", "products.csv", "item_data.csv", "product_data.csv"],
        "ürün bilgileri (items)")

    terms_path = _find_csv(data_dir,
        ["terms.csv", "queries.csv", "search_terms.csv", "query_data.csv"],
        "arama terimleri (terms)")

    train_path = _find_csv(data_dir,
        ["training_pairs.csv", "train.csv", "train_pairs.csv", "training.csv"],
        "eğitim çiftleri (training_pairs)")

    sub_pairs_path = _find_csv(data_dir,
        ["submission_pairs.csv", "test.csv", "test_pairs.csv"],
        "test çiftleri (submission_pairs)")

    sample_sub_path = _find_csv(data_dir,
        ["sample_submission.csv", "sampleSubmission.csv"],
        "örnek gönderim (sample_submission)")

    items       = pd.read_csv(items_path)
    terms       = pd.read_csv(terms_path)
    train_pairs = pd.read_csv(train_path)
    sub_pairs   = pd.read_csv(sub_pairs_path)
    sample_sub  = pd.read_csv(sample_sub_path)

    # Sütun isimlerini yazdır (hata ayıklama için)
    print(f"\n[INFO] items       : {items.shape}  -> sutunlar: {list(items.columns)}")
    print(f"[INFO] terms       : {terms.shape}  -> sutunlar: {list(terms.columns)}")
    print(f"[INFO] train_pairs : {train_pairs.shape}  -> sutunlar: {list(train_pairs.columns)}")
    print(f"[INFO] sub_pairs   : {sub_pairs.shape}  -> sutunlar: {list(sub_pairs.columns)}")
    print(f"[INFO] sample_sub  : {sample_sub.shape}  -> sutunlar: {list(sample_sub.columns)}")

    # ── Sütun isimlerini standartlaştır ──
    # items: 'item_id' veya 'product_id' → 'item_id'
    items = _standardize_columns(items, {
        "product_id": "item_id", "content_id": "item_id",
        "product_name": "title", "name": "title",
        "category_name": "category", "category_path": "category",
        "material": "attributes", "color": "brand"  # fallback
    })

    # terms: 'term_id' ve 'query' → standartlaştır
    terms = _standardize_columns(terms, {
        "search_term": "query", "search_query": "query",
        "query_id": "term_id", "search_id": "term_id",
    })

    # train_pairs: 'id', 'term_id', 'item_id', 'label'
    train_pairs = _standardize_columns(train_pairs, {
        "product_id": "item_id", "content_id": "item_id",
        "query_id": "term_id", "search_id": "term_id",
        "relevance": "label", "target": "label", "is_relevant": "label",
    })

    # label sütunu yoksa ekle (hepsi 1 = pozitif)
    if "label" not in train_pairs.columns:
        train_pairs["label"] = 1
        print("[INFO] Eğitim verisinde 'label' sütunu bulunamadı, hepsi 1 olarak eklendi.")

    # sub_pairs: 'id', 'term_id', 'item_id'
    sub_pairs = _standardize_columns(sub_pairs, {
        "product_id": "item_id", "content_id": "item_id",
        "query_id": "term_id", "search_id": "term_id",
    })

    # sample_sub: 'id', 'prediction'
    sample_sub = _standardize_columns(sample_sub, {
        "target": "prediction", "label": "prediction",
        "is_relevant": "prediction", "relevance": "prediction",
    })

    return items, terms, train_pairs, sub_pairs, sample_sub


def _standardize_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Bilinen alternatif sütun isimlerini standart isimlere çevirir."""
    for old_name, new_name in rename_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. NEGATİF ÖRNEKLEME (Negative Sampling)
# ═══════════════════════════════════════════════════════════════════════════

def create_negative_samples(train_pairs: pd.DataFrame,
                            items: pd.DataFrame,
                            terms: pd.DataFrame) -> pd.DataFrame:
    """
    Eğitim verisi yalnızca label=1 (alakalı) çiftlerden oluşuyor.
    Bu fonksiyon, her pozitif çift için FARKLI bir ürünü eşleştirerek
    negatif (label=0) örnekler üretir. Sonuç: %50 pozitif, %50 negatif.

    Strateji: Her sorgu için pozitif ürünün kategorisinden FARKLI bir
    kategorideki rastgele ürünü seçerek daha güçlü negatifler üretiyoruz.
    """
    print("[INFO] Negatif örnekleme başlıyor...")

    # Pozitif çiftleri items ile birleştir (kategori bilgisi almak için)
    has_category = "category" in items.columns
    if has_category:
        pos_with_cat = train_pairs.merge(items[["item_id", "category"]], on="item_id", how="left")
        item_to_cat = dict(zip(items["item_id"], items["category"].fillna("")))
    else:
        pos_with_cat = train_pairs.copy()
        pos_with_cat["category"] = ""
        item_to_cat = {}
        print("  [UYARI] 'category' sütunu bulunamadı, saf rastgele negatif örnekleme yapılacak.")

    all_item_ids = items["item_id"].values

    neg_records = []
    neg_counter = 0

    # Pozitif çiftlerdeki eşsiz (term_id, item_id) setini oluştur
    positive_set = set(zip(train_pairs["term_id"], train_pairs["item_id"]))

    for _, row in tqdm(pos_with_cat.iterrows(), total=len(pos_with_cat), desc="Negatif ornekleme"):
        term_id = row["term_id"]
        pos_category = str(row["category"]) if pd.notna(row["category"]) else ""

        # Farklı bir ürün seç — tercihen farklı kategoriden
        max_attempts = 20
        for _ in range(max_attempts):
            random_item = np.random.choice(all_item_ids)
            random_cat = item_to_cat.get(random_item, "")

            # Zaten pozitif bir çift değilse VE farklı kategorideyse kabul et
            if (term_id, random_item) not in positive_set:
                # Kategori farklıysa ideal; aynıysa da en fazla max_attempts dene
                if random_cat != pos_category or _ == max_attempts - 1:
                    break

        neg_records.append({
            "id": f"NEG_{neg_counter:08d}",
            "term_id": term_id,
            "item_id": random_item,
            "label": 0
        })
        neg_counter += 1

    neg_df = pd.DataFrame(neg_records)
    balanced = pd.concat([train_pairs, neg_df], ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(f"[INFO] Pozitif sayısı : {(balanced['label'] == 1).sum()}")
    print(f"[INFO] Negatif sayısı : {(balanced['label'] == 0).sum()}")
    print(f"[INFO] Toplam eğitim  : {len(balanced)}")
    return balanced


# ═══════════════════════════════════════════════════════════════════════════
# 3. METİN ÖN İŞLEME & ÖZELLİK MÜHENDİSLİĞİ
# ═══════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Küçük harfe çevir, noktalama temizle, fazla boşlukları sil."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\sçğıöşüâîû]", " ", text)    # Türkçe karakterleri koru
    text = re.sub(r"\s+", " ", text).strip()
    return text


def merge_and_clean(pairs: pd.DataFrame,
                    terms: pd.DataFrame,
                    items: pd.DataFrame) -> pd.DataFrame:
    """Çift verisini terms ve items ile birleştirip metin temizliği uygular."""
    df = pairs.merge(terms, on="term_id", how="left")
    df = df.merge(items, on="item_id", how="left")

    # Beklenen ama veri setinde olmayabilecek sütunları boş oluştur
    expected_text_cols = ["query", "title", "category", "brand", "gender", "age_group", "attributes"]
    for col in expected_text_cols:
        if col not in df.columns:
            df[col] = ""
            print(f"  [UYARI] '{col}' sütunu veri setinde bulunamadı, boş olarak oluşturuldu.")

    # NaN dolgusu
    for col in expected_text_cols:
        df[col] = df[col].fillna("")

    # Metin temizleme
    tqdm.pandas(desc="Temizleniyor: query")
    df["query_clean"]      = df["query"].progress_apply(clean_text)
    tqdm.pandas(desc="Temizleniyor: title")
    df["title_clean"]      = df["title"].progress_apply(clean_text)
    tqdm.pandas(desc="Temizleniyor: category")
    df["category_clean"]   = df["category"].progress_apply(clean_text)
    tqdm.pandas(desc="Temizleniyor: brand")
    df["brand_clean"]      = df["brand"].progress_apply(clean_text)
    tqdm.pandas(desc="Temizleniyor: attributes")
    df["attributes_clean"] = df["attributes"].progress_apply(clean_text)

    # Ürün detaylarını tek bir metin alanında birleştir (TF-IDF için)
    tqdm.pandas(desc="Urun detaylari birlestiriliyor")
    df["product_text"] = (
        df["title_clean"] + " " +
        df["category_clean"] + " " +
        df["brand_clean"] + " " +
        df["attributes_clean"]
    ).progress_apply(lambda x: re.sub(r"\s+", " ", x).strip())

    return df


def build_tfidf_cosine(df: pd.DataFrame,
                       query_col: str = "query_clean",
                       product_col: str = "product_text",
                       max_features: int = 50_000) -> np.ndarray:
    """
    TF-IDF vektörlerini oluştur ve Kosinüs Benzerliğini hesapla.
    Hem eğitim hem de test verisi için kullanılacak.
    """
    print(f"    TF-IDF fit ediliyor ({len(df)} satir)...")
    tfidf = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2
    )

    all_texts = pd.concat([df[query_col], df[product_col]], ignore_index=True)
    tfidf.fit(all_texts)

    print("    Query vektorleri olusturuluyor...")
    query_vecs   = tfidf.transform(df[query_col])
    print("    Product vektorleri olusturuluyor...")
    product_vecs = tfidf.transform(df[product_col])

    # Vektorize kosinus benzerligi (sparse matrix multiply - cok hizli)
    print("    Kosinus benzerligi hesaplaniyor (vektorize)...")
    from sklearn.preprocessing import normalize
    query_norm = normalize(query_vecs, norm='l2')
    product_norm = normalize(product_vecs, norm='l2')
    cos_sims = np.array(query_norm.multiply(product_norm).sum(axis=1)).flatten()
    print(f"    Tamamlandi! ({len(cos_sims)} skor)")
    return cos_sims


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sayısal özellik vektörlerini oluşturur."""
    print(f"[INFO] Ozellik muhendisligi basliyor ({len(df)} satir)...")

    features = pd.DataFrame(index=df.index)

    # ── 1. TF-IDF Kosinüs Benzerliği ──
    print("  > TF-IDF kosinus benzerligi hesaplaniyor...")
    features["cos_sim_query_product"] = build_tfidf_cosine(df)

    # ── 2. Kelime Kesişim Özellikleri ──
    tqdm.pandas(desc="Sorgu kelimeleri (set)")
    q_words = df["query_clean"].progress_apply(lambda x: set(x.split()))
    tqdm.pandas(desc="Urun kelimeleri (set)")
    p_words = df["product_text"].progress_apply(lambda x: set(x.split()))
    tqdm.pandas(desc="Baslik kelimeleri (set)")
    t_words = df["title_clean"].progress_apply(lambda x: set(x.split()))

    # Sorgu <-> urun detayi kelime kesisimi
    features["word_overlap_count"] = [
        len(q & p) for q, p in tqdm(zip(q_words, p_words), total=len(df), desc="Ozellik: word_overlap_count")
    ]
    features["word_overlap_ratio"] = [
        len(q & p) / max(len(q), 1) for q, p in tqdm(zip(q_words, p_words), total=len(df), desc="Ozellik: word_overlap_ratio")
    ]

    # Sorgu <-> baslik kelime kesisimi
    features["title_overlap_count"] = [
        len(q & t) for q, t in tqdm(zip(q_words, t_words), total=len(df), desc="Ozellik: title_overlap_count")
    ]
    features["title_overlap_ratio"] = [
        len(q & t) / max(len(q), 1) for q, t in tqdm(zip(q_words, t_words), total=len(df), desc="Ozellik: title_overlap_ratio")
    ]

    # ── 3. Uzunluk Farkı Özellikleri ──
    features["query_len_char"]    = df["query_clean"].str.len()
    features["title_len_char"]    = df["title_clean"].str.len()
    features["product_len_char"]  = df["product_text"].str.len()
    features["query_word_count"]  = df["query_clean"].str.split().str.len()
    features["title_word_count"]  = df["title_clean"].str.split().str.len()
    features["len_diff_char"]     = abs(features["query_len_char"] - features["title_len_char"])
    features["len_diff_word"]     = abs(features["query_word_count"] - features["title_word_count"])
    features["len_ratio"]         = features["query_len_char"] / (features["title_len_char"] + 1)

    # ── 4. Kategori Ağacı Eşleşme Özellikleri ──
    # Kategori hiyerarşisi "/" ile ayrılıyor (ör: ayakkabı/spor ayakkabı/sneaker)
    tqdm.pandas(desc="Kategori agaci parcalaniyor")
    cat_levels = df["category_clean"].progress_apply(lambda x: x.split("/") if x else [])

    # Sorgu kelimelerinin kategori seviyelerine düşmesi
    features["cat_match_any"] = [
        int(any(w in "/".join(cats) for w in q)) if q and cats else 0
        for q, cats in tqdm(zip(q_words, cat_levels), total=len(df), desc="Ozellik: cat_match_any")
    ]
    features["cat_match_count"] = [
        sum(1 for w in q if w in "/".join(cats)) if q and cats else 0
        for q, cats in tqdm(zip(q_words, cat_levels), total=len(df), desc="Ozellik: cat_match_count")
    ]
    features["cat_depth"] = cat_levels.apply(len)

    # ── 5. Marka Eşleşmesi ──
    features["brand_in_query"] = [
        int(b.strip() != "" and b in q) if b and q else 0
        for q, b in tqdm(zip(df["query_clean"], df["brand_clean"]), total=len(df), desc="Ozellik: brand_in_query")
    ]

    # ── 6. Cinsiyet / Yaş Grubu Eşleşmesi ──
    features["gender_in_query"] = [
        int(g.strip() != "" and g != "unknown" and g in q)
        for q, g in tqdm(zip(df["query_clean"], df["gender"].fillna("unknown").str.lower()), total=len(df), desc="Ozellik: gender_in_query")
    ]
    features["age_group_in_query"] = [
        int(a.strip() != "" and a != "unknown" and a in q)
        for q, a in tqdm(zip(df["query_clean"], df["age_group"].fillna("unknown").str.lower()), total=len(df), desc="Ozellik: age_group_in_query")
    ]

    # ── 7. Sorgu kelimelerinin başlıkta tam geçme oranı ──
    features["exact_query_in_title"] = [
        int(q in t) if q else 0
        for q, t in tqdm(zip(df["query_clean"], df["title_clean"]), total=len(df), desc="Ozellik: exact_query_in_title")
    ]

    print(f"[INFO] Toplam özellik sayısı: {features.shape[1]}")
    return features


# ═══════════════════════════════════════════════════════════════════════════
# 4. MODEL EĞİTİMİ & LOKAL DOĞRULAMA (Stratified 5-Fold)
# ═══════════════════════════════════════════════════════════════════════════

def train_and_validate(X: pd.DataFrame,
                       y: pd.Series,
                       n_folds: int = 5) -> list:
    """
    Stratified K-Fold ile CatBoost eğitimi yapar.
    Her fold için Macro-F1 skorunu yazdırır.
    Eğitilmiş modelleri döndürür.
    """
    print(f"\n{'='*60}")
    print(f"  MODEL EGITIMI - Stratified {n_folds}-Fold Cross Validation")
    print(f"{'='*60}\n")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5,
            random_seed=RANDOM_STATE,
            eval_metric="F1",
            auto_class_weights="Balanced",
            verbose=100,
            early_stopping_rounds=100,
            task_type="GPU",          # GPU varsa "GPU" olarak değiştirilebilir
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )

        val_preds = model.predict(X_val).astype(int)
        macro_f1 = f1_score(y_val, val_preds, average="macro")
        fold_scores.append(macro_f1)
        models.append(model)

        print(f"\n  * Fold {fold}/{n_folds} -> Macro-F1: {macro_f1:.5f}\n")

    mean_f1 = np.mean(fold_scores)
    std_f1  = np.std(fold_scores)
    print(f"\n{'-'*60}")
    print(f"  Ortalama Macro-F1 : {mean_f1:.5f} +/- {std_f1:.5f}")
    print(f"{'-'*60}\n")

    return models


# ═══════════════════════════════════════════════════════════════════════════
# 5. TAHMİN & SUBMISSION DOSYASI ÜRETİMİ
# ═══════════════════════════════════════════════════════════════════════════

def predict_and_submit(models: list,
                       X_test: pd.DataFrame,
                       sub_pairs: pd.DataFrame,
                       sample_sub: pd.DataFrame):
    """
    Tüm fold modellerinin tahminlerini çoğunluk oylamasıyla (majority voting)
    birleştirir ve submission.csv dosyası üretir.
    """
    print("[INFO] Test seti tahminleri üretiliyor...")

    # Her modelden tahmin al
    all_preds = np.zeros((len(X_test), len(models)))
    for i, model in enumerate(models):
        all_preds[:, i] = model.predict(X_test).astype(int)

    # Çoğunluk oylaması (majority voting)
    final_preds = (all_preds.mean(axis=1) >= 0.5).astype(int)

    # submission.csv oluştur — sample_submission formatına uygun
    submission = pd.DataFrame({
        "id": sub_pairs["id"].values,
        "prediction": final_preds
    })

    # sample_submission ile sıralama tutarlılığını garanti et
    submission = submission.set_index("id").reindex(sample_sub["id"]).reset_index()
    submission["prediction"] = submission["prediction"].fillna(1).astype(int)

    output_path = "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"[INFO] Submission dosyası kaydedildi: {output_path}")
    print(f"[INFO] Submission boyutu: {submission.shape}")
    print(f"[INFO] Tahmin dağılımı:\n{submission['prediction'].value_counts()}")

    return submission


# ═══════════════════════════════════════════════════════════════════════════
# ANA ÇALIŞMA AKIŞI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Trendyol Datathon 2026 – Baseline CatBoost Pipeline")
    print("=" * 60)

    # ── Adım 1: Veri Okuma ──
    items, terms, train_pairs, sub_pairs, sample_sub = load_data()

    # ── Adım 2: Negatif Örnekleme ──
    balanced_train = create_negative_samples(train_pairs, items, terms)

    # ── Adım 3: Metin Ön İşleme & Birleştirme ──
    print("\n[INFO] Eğitim verisi birleştiriliyor ve temizleniyor...")
    train_df = merge_and_clean(balanced_train, terms, items)

    print("[INFO] Test verisi birleştiriliyor ve temizleniyor...")
    test_df = merge_and_clean(sub_pairs, terms, items)

    # ── Adım 3b: Özellik Çıkarımı ──
    # ÖNEMLİ: TF-IDF'yi eğitim+test birleşik corpus üzerinde fit edelim
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    print("\n[INFO] Birleşik TF-IDF kosinüs benzerliği hesaplanıyor...")
    combined_cos_sims = build_tfidf_cosine(combined_df)
    train_df["cos_sim_combined"] = combined_cos_sims[:len(train_df)]
    test_df["cos_sim_combined"]  = combined_cos_sims[len(train_df):]

    # Eğitim özellikleri
    print("\n[INFO] Eğitim seti özellikleri çıkarılıyor...")
    X_train = extract_features(train_df)
    X_train["cos_sim_combined"] = train_df["cos_sim_combined"].values
    y_train = train_df["label"]

    # Test özellikleri
    print("[INFO] Test seti özellikleri çıkarılıyor...")
    X_test = extract_features(test_df)
    X_test["cos_sim_combined"] = test_df["cos_sim_combined"].values

    # Sütun sıralamasını eşitle
    X_test = X_test[X_train.columns]

    # ── Adım 4: Model Eğitimi & Doğrulama ──
    models = train_and_validate(X_train, y_train, n_folds=5)

    # ── Adım 5: Tahmin & Submission ──
    submission = predict_and_submit(models, X_test, sub_pairs, sample_sub)

    # ── Adım 6: Kaggle CLI ile Otomatik Gönderim ──
    print("\n[INFO] Kaggle CLI ile submission gönderiliyor...")
    submit_cmd = (
        'kaggle competitions submit '
        '-c trendyol-e-ticaret-yarismasi-2026-kaggle '
        '-f submission.csv '
        '-m "Baseline CatBoost with Kagglehub Path and Negative Sampling"'
    )
    exit_code = os.system(submit_cmd)
    if exit_code == 0:
        print("[INFO] Submission basariyla gonderildi!")
    else:
        print(f"[UYARI] Kaggle CLI gonderimi basarisiz (exit code: {exit_code}).")
        print("        Kaggle CLI kurulu ve kimlik bilgileriniz dogru mu kontrol edin.")
        print("        Manuel gonderim: kaggle competitions submit -c trendyol-e-ticaret-yarismasi-2026-kaggle -f submission.csv -m 'manual'")

    print("\n" + "=" * 60)
    print("  Pipeline tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
