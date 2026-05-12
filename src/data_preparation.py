import os
import shutil
import kagglehub
from src.config import RAW_DIR, PROCESSED_DIR


CATEGORY_MAP = {
    "Closed_Eyes": "Closed",
    "Open_Eyes": "Open",
    "Yawn": "yawn",
    "No_yawn": "no_yawn",
    "closed": "Closed",
    "open": "Open",
    "yawn": "yawn",
    "no_yawn": "no_yawn",
}


def download_drowsiness_dataset():
    print("Baixando Drowsiness Dataset do Kaggle...")
    path = kagglehub.dataset_download("hoangtung719/drowsiness-dataset")
    print(f"Dataset baixado em: {path}")
    return path


def download_mrl_dataset():
    print("Baixando MRL Eye Dataset do Kaggle...")
    path = kagglehub.dataset_download("prashantsuravajjala/mrl-eye-dataset")
    print(f"Dataset baixado em: {path}")
    return path


def _resolve_category(dirname):
    lower = dirname.lower()
    for key, val in CATEGORY_MAP.items():
        if key.lower() == lower:
            return val
    if "closed" in lower or "close" in lower:
        return "Closed"
    if "open" in lower:
        return "Open"
    if "yawn" in lower and "no" not in lower:
        return "yawn"
    if "no" in lower and "yawn" in lower:
        return "no_yawn"
    return None


def _find_split_dirs(source_path):
    for split_name in ["Train", "train", "Training"]:
        d = os.path.join(source_path, split_name)
        if os.path.isdir(d):
            return d, None
    for split_name in ["Dataset/Train", "dataset/train"]:
        d = os.path.join(source_path, split_name)
        if os.path.isdir(d):
            v = os.path.join(source_path, split_name.replace("Train", "Val").replace("train", "val"))
            v2 = os.path.join(source_path, split_name.replace("Train", "Test").replace("train", "test"))
            val_d = v if os.path.isdir(v) else (v2 if os.path.isdir(v2) else None)
            return d, val_d
    return None, None


def organize_drowsiness_dataset(source_path):
    train_dir = os.path.join(PROCESSED_DIR, "train")
    val_dir = os.path.join(PROCESSED_DIR, "val")
    test_dir = os.path.join(PROCESSED_DIR, "test")

    for d in [train_dir, val_dir, test_dir]:
        for cat in ["Closed", "Open", "yawn", "no_yawn"]:
            os.makedirs(os.path.join(d, cat), exist_ok=True)

    src_train, src_val = _find_split_dirs(source_path)
    print(f"  Fonte treino: {src_train}")
    print(f"  Fonte validação: {src_val}")

    src_test = None
    for test_name in ["Test", "test", "Dataset/Test", "dataset/test"]:
        t = os.path.join(source_path, test_name)
        if os.path.isdir(t):
            src_test = t
            break

    def _copy_category(src_split_dir, dst_split_dir):
        if src_split_dir is None:
            return
        for entry in os.listdir(src_split_dir):
            entry_path = os.path.join(src_split_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            category = _resolve_category(entry)
            if category is None:
                print(f"    [SKIP] Categoria não reconhecida: {entry}")
                continue
            files = [f for f in os.listdir(entry_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            dst_cat = os.path.join(dst_split_dir, category)
            for f in files:
                shutil.copy2(os.path.join(entry_path, f), os.path.join(dst_cat, f))
            print(f"    {entry} -> {category}: {len(files)} arquivos")

    print("  Copiando treino...")
    _copy_category(src_train, train_dir)

    print("  Copiando validação...")
    if src_val:
        _copy_category(src_val, val_dir)
    else:
        print("    Sem split de validação. Dividindo treino (80/20)...")
        for cat in ["Closed", "Open", "yawn", "no_yawn"]:
            cat_train = os.path.join(train_dir, cat)
            cat_val = os.path.join(val_dir, cat)
            files = [f for f in os.listdir(cat_train)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            val_count = int(len(files) * 0.2)
            for f in files[:val_count]:
                shutil.move(os.path.join(cat_train, f), os.path.join(cat_val, f))
            print(f"    {cat}: {len(files) - val_count} treino, {val_count} validação")

    print("  Copiando teste...")
    if src_test:
        _copy_category(src_test, test_dir)

    for split_name, split_dir in [("Treino", train_dir), ("Validação", val_dir), ("Teste", test_dir)]:
        print(f"\n  {split_name}:")
        for cat in ["Closed", "Open", "yawn", "no_yawn"]:
            count = len([f for f in os.listdir(os.path.join(split_dir, cat))
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"    {cat}: {count}")

    print(f"\n  Dataset organizado em {PROCESSED_DIR}")
    return train_dir, val_dir, test_dir


def prepare_dataset(dataset_choice="drowsiness"):
    if dataset_choice == "drowsiness":
        source = download_drowsiness_dataset()
        return organize_drowsiness_dataset(source)
    elif dataset_choice == "mrl":
        source = download_mrl_dataset()
        return organize_mrl_dataset(source)
    elif dataset_choice == "both":
        source1 = download_drowsiness_dataset()
        result = organize_drowsiness_dataset(source1)
        source2 = download_mrl_dataset()
        organize_mrl_dataset(source2)
        return result
    else:
        raise ValueError(f"Dataset desconhecido: {dataset_choice}")


def organize_mrl_dataset(source_path):
    train_dir = os.path.join(PROCESSED_DIR, "train")
    val_dir = os.path.join(PROCESSED_DIR, "val")

    os.makedirs(os.path.join(train_dir, "Closed"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "Open"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "Closed"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "Open"), exist_ok=True)

    for root, dirs, files in os.walk(source_path):
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            filepath = os.path.join(root, f)
            try:
                state = int(f.split('_')[4])
            except (IndexError, ValueError):
                continue

            label = "Closed" if state == 1 else "Open"
            files_count = len(os.listdir(os.path.join(train_dir, label)))
            dst_dir = train_dir if files_count < 3000 else val_dir
            shutil.copy2(filepath, os.path.join(dst_dir, label, f))

    print("MRL Eye Dataset organizado com sucesso.")


if __name__ == "__main__":
    import sys
    choice = sys.argv[1] if len(sys.argv) > 1 else "drowsiness"
    prepare_dataset(choice)
