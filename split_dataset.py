import os, shutil, random

def split(src_dir, dst_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    random.seed(seed)
    for label in os.listdir(src_dir):
        label_path = os.path.join(src_dir, label)
        if not os.path.isdir(label_path): continue
        files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg','.png'))]
        random.shuffle(files)
        n = len(files)
        t = int(n*train_ratio)
        v = int(n*(train_ratio+val_ratio))
        parts = {'train': files[:t], 'val': files[t:v], 'test': files[v:]}
        for part, items in parts.items():
            target_dir = os.path.join(dst_dir, part, label)
            os.makedirs(target_dir, exist_ok=True)
            for fname in items:
                shutil.copy(os.path.join(label_path,fname), os.path.join(target_dir,fname))

if __name__ == "__main__":
    split("raw_dataset", "dataset")
