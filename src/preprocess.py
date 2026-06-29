import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

CLASS_TO_IDX = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}

def main():
    metadata_csv = os.environ.get('HAM10000_METADATA_CSV', 'c:/DL/Project/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
    images_dir = os.environ.get('HAM10000_IMAGES_DIR', 'c:/DL/Project/skin-cancer-mnist-ham10000')
    
    print(f"Loading metadata from: {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    
    # Create image paths dictionary
    image_paths = {}
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
        for path in glob.glob(os.path.join(images_dir, '**', ext), recursive=True):
            img_id = os.path.splitext(os.path.basename(path))[0]
            image_paths[img_id] = path
            
    df['image_path'] = df['image_id'].map(image_paths)
    
    # Drop rows without image path
    missing_count = df['image_path'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} images in metadata not found in folders.")
        df = df.dropna(subset=['image_path'])
        
    df['label_idx'] = df['dx'].map(CLASS_TO_IDX)
    
    # Merge with supplementary metadata if exists
    supp_metadata_csv = 'data/isic_supplementary_metadata.csv'
    if os.path.exists(supp_metadata_csv):
        print(f"Loading supplementary metadata from: {supp_metadata_csv}")
        df_supp = pd.read_csv(supp_metadata_csv)
        df_supp['image_path'] = df_supp['image_path'].apply(os.path.abspath)
        print(f"Loaded {len(df_supp)} supplementary images.")
        df = pd.concat([df, df_supp], ignore_index=True)
        
    print(f"Total samples matched: {len(df)}")
    print("Class distribution:")
    class_counts = df['dx'].value_counts()
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt} ({cnt/len(df)*100:.2f}%)")
        
    # Generate and save class distribution plot
    os.makedirs('outputs/figures', exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='dx', order=class_counts.index, palette='viridis')
    plt.title('HAM10000 Class Distribution')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('outputs/figures/class_distribution.png')
    plt.close()
    print("Saved class distribution plot to outputs/figures/class_distribution.png")
    
    # Stratified group split: Group by lesion_id to prevent data leakage (same lesion in train and test)
    # while maintaining stratified distribution of classes.
    lesion_df = df.groupby('lesion_id').first().reset_index()
    
    train_lesions, temp_lesions = train_test_split(
        lesion_df, test_size=0.30, random_state=42, stratify=lesion_df['label_idx']
    )
    dev_lesions, test_lesions = train_test_split(
        temp_lesions, test_size=0.50, random_state=42, stratify=temp_lesions['label_idx']
    )
    
    # Map back to image-level splits
    train_df = df[df['lesion_id'].isin(train_lesions['lesion_id'])]
    dev_df = df[df['lesion_id'].isin(dev_lesions['lesion_id'])]
    test_df = df[df['lesion_id'].isin(test_lesions['lesion_id'])]
    
    # Save splits
    os.makedirs('data/splits', exist_ok=True)
    train_df.to_csv('data/splits/train.csv', index=False)
    dev_df.to_csv('data/splits/dev.csv', index=False)
    test_df.to_csv('data/splits/test.csv', index=False)
    print("Saved train.csv, dev.csv, and test.csv to data/splits/")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val (dev) samples: {len(dev_df)}")
    print(f"  Test samples: {len(test_df)}")


if __name__ == '__main__':
    main()
