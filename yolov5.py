import os
import shutil
import scipy.io
from sklearn.model_selection import train_test_split
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import tarfile
import subprocess

working_directory = '/home/ubuntu/alon'

images_path = f"{working_directory}/102flowers.tgz"
extracted_folder = f"{working_directory}/102flowers/"

with tarfile.open(images_path, "r:gz") as tar:
    tar.extractall(path=extracted_folder)

print("Extraction complete!")

images = []
for root, dirs, files in os.walk(extracted_folder):
    for f in files:
        images.append(os.path.join(root, f))

print(f"Total images found: {len(images)}")

label_path = f"{working_directory}/imagelabels.mat"

splits = []
for i in range(2):
    train_images, test_images = train_test_split(images, test_size=0.5, random_state=42 + i)
    val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42 + i)

    print(f"Training images (split {i + 1}): {len(train_images)}")
    print(f"Validation images (split {i + 1}): {len(val_images)}")
    print(f"Test images (split {i + 1}): {len(test_images)}")

    splits.append((train_images, val_images, test_images))

labels_data = scipy.io.loadmat(label_path)
labels = labels_data['labels'][0]

def copy_images(image_paths, dest_dir):
    for img_path in image_paths:
        shutil.copy(img_path, dest_dir)

def create_label_files(image_paths, labels, output_dir):
    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        class_id = int(labels[i]) - 1
        label_path = os.path.join(output_dir, label_name)
        with open(label_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

def create_yaml(train_dir, val_dir, nc=102):
    data_yaml = {
        'train': train_dir,
        'val': val_dir,
        'test': test_dir,
        'nc': nc,
        'names': ['flower_{}'.format(i) for i in range(nc)]
    }
    return data_yaml

def clone_yolov5():
    """Clone the YOLOv5 repository."""
    if not os.path.exists("yolov5"):
        subprocess.check_call(["git", "clone", "https://github.com/ultralytics/yolov5.git"])

def train_yolov5(i, data_yaml_path, model_weights="yolov5s.pt", epochs=50):
    """Train a YOLOv5 model on a given dataset."""
    clone_yolov5()

    os.chdir("yolov5")

    command = [
        "python", "train.py",
        "--img", "640",
        "--batch", "32",
        "--epochs", str(epochs),
        "--data", data_yaml_path,
        "--weights", model_weights,
        "--device", "0",
        "--project", "flower_classification",
        "--name", f"run_{i}"
    ]

    subprocess.check_call(command)

def validate_yolov5(i, data_yaml_path, weights_path, output_dir="flower_classification"):
    """Validate a YOLOv5 model on a given dataset after training."""

    os.chdir("yolov5")

    command = [
        "python", "val.py",
        "--img", "640",
        "--data", data_yaml_path,
        "--weights", weights_path,
        "--device", "0",
        "--project", output_dir,
        "--name", f"run_{i}",
        "--task", "test"
    ]
    subprocess.check_call(command)

def plot_results(train_log, output_file='results.png'):
    df = pd.read_csv(train_log)
    df.columns = df.columns.str.strip()

    train_accuracy = df['metrics/precision']
    val_accuracy = df['metrics/mAP_0.5']
    try:
        test_accuracy = df['test/mAP_0.5']
    except KeyError:
        test_accuracy = None

    train_loss = df['train/box_loss'] + df['train/obj_loss'] + df['train/cls_loss']
    val_loss = df['val/box_loss'] + df['val/obj_loss'] + df['val/cls_loss']
    try:
        test_loss = df['test/box_loss'] + df['test/obj_loss'] + df['test/cls_loss']
    except KeyError:
        test_loss = None

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(df['epoch'], train_accuracy, label='Train Accuracy', color='blue')
    axs[0].plot(df['epoch'], val_accuracy, label='Validation Accuracy (mAP@0.5)', color='orange')
    if test_accuracy is not None:
        axs[0].plot(df['epoch'], test_accuracy, label='Test Accuracy (mAP@0.5)', color='green')
    axs[0].set_title('Accuracy over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy (mAP@0.5 or Precision)')
    axs[0].legend()

    axs[1].plot(df['epoch'], train_loss, label='Train Loss', color='blue')
    axs[1].plot(df['epoch'], val_loss, label='Validation Loss', color='orange')
    if test_loss is not None:
        axs[1].plot(df['epoch'], test_loss, label='Test Loss', color='green')
    axs[1].set_title('Loss over Epochs (Cross-Entropy)')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()

    plt.savefig(output_file)

    plt.show()

for i, (train_images, val_images, test_images) in enumerate(splits):
    train_dir = f"{working_directory}/train_images_{i}"
    val_dir = f"{working_directory}/val_images_{i}"
    test_dir = f"{working_directory}/test_images_{i}"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    copy_images(train_images, train_dir)
    copy_images(val_images, val_dir)
    copy_images(test_images, test_dir)

    create_label_files(train_images, labels[:len(train_images)], train_dir)
    create_label_files(val_images, labels[len(train_images):len(train_images) + len(val_images)], val_dir)
    create_label_files(test_images, labels[len(train_images) + len(val_images):], test_dir)

    data_yaml = create_yaml(train_dir, val_dir)
    yaml_path = f"{working_directory}/data_{i}.yaml"
    with open(yaml_path, 'w') as file:
        yaml.dump(data_yaml, file)

    train_yolov5(i, yaml_path, model_weights="yolov5s.pt", epochs=50)

    weights_path = f"{working_directory}/yolov5/flower_classification/run_{i}/weights/best.pt"
    validate_yolov5(i, yaml_path, weights_path)

    train_log = f"{working_directory}/yolov5/flower_classification/run_{i}/results.csv"
    plot_results(train_log, f"results_{i}.png")