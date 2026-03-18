from pathlib import Path
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 45
VAL_SPLIT = 0.2
SEED = 42
DATASET_DIR = Path("processed_dataset")
MODEL_PATH = "autism_mri_model.h5"
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "processed_dataset"
MODEL_PATH = str(BASE_DIR / "autism_mri_model.h5")

CLASS_TO_LABEL = {
    "autistic": 1,
    "non_autistic": 0,
}


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def extract_subject_id(file_path: Path) -> str:
    file_name = file_path.name
    if ".nii_" in file_name:
        return file_name.split(".nii_")[0]
    return file_path.stem


def collect_samples(dataset_dir: Path):
    samples = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue
        for file_path in class_dir.glob("*.png"):
            samples.append((str(file_path), label, extract_subject_id(file_path)))
    if not samples:
        raise ValueError("No PNG files found in processed_dataset. Run create_dataset.py first.")
    return samples


def split_by_subject(samples, val_split: float, seed: int):
    rng = np.random.default_rng(seed)

    train_items = []
    val_items = []

    for label in (0, 1):
        class_items = [item for item in samples if item[1] == label]
        subject_ids = np.array(sorted({item[2] for item in class_items}))

        if len(subject_ids) < 2:
            raise ValueError(
                f"Not enough subjects in class {label} for train/validation split."
            )

        rng.shuffle(subject_ids)
        val_count = max(1, int(round(len(subject_ids) * val_split)))
        val_subjects = set(subject_ids[:val_count])

        for item in class_items:
            if item[2] in val_subjects:
                val_items.append(item)
            else:
                train_items.append(item)

    rng.shuffle(train_items)
    rng.shuffle(val_items)

    return train_items, val_items


def build_dataset(items, training: bool):
    paths = [item[0] for item in items]
    labels = [item[1] for item in items]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        return image, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model() -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode="horizontal"),
            tf.keras.layers.RandomRotation(factor=0.08),
            tf.keras.layers.RandomZoom(height_factor=0.12, width_factor=0.12),
            tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
            tf.keras.layers.RandomContrast(factor=0.12),
        ],
        name="augmentation",
    )

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = data_augmentation(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(96, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(2e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    set_seeds(SEED)

    samples = collect_samples(DATASET_DIR)
    train_items, val_items = split_by_subject(samples, VAL_SPLIT, SEED)

    train_subjects = {item[2] for item in train_items}
    val_subjects = {item[2] for item in val_items}
    overlap = train_subjects.intersection(val_subjects)

    print(f"Train images: {len(train_items)}")
    print(f"Val images:   {len(val_items)}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects:   {len(val_subjects)}")
    print(f"Subject overlap between train/val: {len(overlap)}")

    train_ds = build_dataset(train_items, training=True)
    val_ds = build_dataset(val_items, training=False)

    model = build_model()
    model.summary()

    training_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=training_callbacks,
        verbose=1,
    )

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

    eval_loss, eval_acc, eval_auc = model.evaluate(val_ds, verbose=0)

    print(f"Final Train Accuracy: {final_train_acc:.4f}")
    print(f"Final Val Accuracy:   {final_val_acc:.4f}")
    print(f"Best Val Accuracy:    {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Evaluated Val Loss:   {eval_loss:.4f}")
    print(f"Evaluated Val Accuracy (best weights): {eval_acc:.4f}")
    print(f"Evaluated Val AUC:    {eval_auc:.4f}")
    print(f"Best model saved at:  {MODEL_PATH}")


if __name__ == "__main__":
    main()