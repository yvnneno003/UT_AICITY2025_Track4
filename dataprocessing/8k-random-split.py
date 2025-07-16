import os
import shutil
from sklearn.model_selection import train_test_split


image_dir = 'datasets/8k/images'
label_dir = 'datasets/8k/labels'
train_image_dir = 'datasets/8k/train/images'
val_image_dir = 'datasets/8k/val/images'
train_label_dir = 'datasets/8k/train/labels'
val_label_dir = 'datasets/8k/val/labels'


os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)


image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]


train_files, val_files = train_test_split(image_files, test_size=0.3, random_state=42)


def move_files_and_labels(file_list, target_image_dir, target_label_dir):
    for file_name in file_list:

        src_image = os.path.join(image_dir, file_name)
        dst_image = os.path.join(target_image_dir, file_name)
        shutil.move(src_image, dst_image)
        

        label_file = file_name.rsplit('.', 1)[0] + '.txt'
        src_label = os.path.join(label_dir, label_file)
        if os.path.isfile(src_label):
            dst_label = os.path.join(target_label_dir, label_file)
            shutil.move(src_label, dst_label)


move_files_and_labels(train_files, train_image_dir, train_label_dir)
move_files_and_labels(val_files, val_image_dir, val_label_dir)

print(f'Total images: {len(image_files)}')
print(f'Training images: {len(train_files)}')
print(f'Validation images: {len(val_files)}')
