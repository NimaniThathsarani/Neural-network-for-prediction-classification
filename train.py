import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# 1. Load & prepare data

df = pd.read_csv('recruitment.csv')

X = df[['Position', 'Experience', 'Tech_Score', 'Interview']]
y_raw = df['Status']

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
n_classes = len(class_names)
print('Classes:', class_names)

# 60 / 20 / 20 train / val / test split, stratified
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

print('Train/Val/Test sizes:', len(X_train), len(X_val), len(X_test))

numeric_features = ['Experience', 'Tech_Score', 'Interview']
categorical_features = ['Position']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train_p = preprocessor.fit_transform(X_train)
X_val_p = preprocessor.transform(X_val)
X_test_p = preprocessor.transform(X_test)

input_dim = X_train_p.shape[1]
print('Input dimension after preprocessing:', input_dim)

y_train_oh = keras.utils.to_categorical(y_train, n_classes)
y_val_oh = keras.utils.to_categorical(y_val, n_classes)
y_test_oh = keras.utils.to_categorical(y_test, n_classes)


# 2. Architecture experiments

def build_model(architecture, dropout=0.0, l2=0.0, lr=1e-3):
    reg = regularizers.l2(l2) if l2 > 0 else None
    model = keras.Sequential(name=architecture['name'])
    model.add(layers.Input(shape=(input_dim,)))
    for units in architecture['hidden']:
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=reg))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_classes, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

architectures = [
    {'name': 'small_1layer',   'hidden': [8]},
    {'name': 'medium_2layer',  'hidden': [32, 16]},
    {'name': 'large_3layer',   'hidden': [64, 32, 16]},
]

results = {}
histories = {}

for arch in architectures:
    model = build_model(arch, dropout=0.0, l2=0.0)
    hist = model.fit(X_train_p, y_train_oh,
                      validation_data=(X_val_p, y_val_oh),
                      epochs=150, batch_size=16, verbose=0)
    val_acc = max(hist.history['val_accuracy'])
    train_acc_final = hist.history['accuracy'][-1]
    val_acc_final = hist.history['val_accuracy'][-1]
    gap = train_acc_final - val_acc_final
    results[arch['name']] = {
        'best_val_acc': val_acc,
        'final_train_acc': train_acc_final,
        'final_val_acc': val_acc_final,
        'overfit_gap': gap
    }
    histories[arch['name']] = hist.history
    print(f"{arch['name']}: best_val_acc={val_acc:.3f}, "
          f"final_train_acc={train_acc_final:.3f}, final_val_acc={val_acc_final:.3f}, gap={gap:.3f}")

# Plot architecture comparison (val accuracy curves)
plt.figure(figsize=(7, 5))
for name, h in histories.items():
    plt.plot(h['val_accuracy'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Validation accuracy')
plt.title('Architecture comparison: validation accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('architectures_comparison.png', dpi=150)
plt.close()


# 3. Overfitting prevention

chosen_arch = {'name': 'medium_2layer', 'hidden': [32, 16]}

# 3a. Plain (no regularization) - reference "overfitting" run, more epochs
plain_model = build_model(chosen_arch, dropout=0.0, l2=0.0)
plain_hist = plain_model.fit(X_train_p, y_train_oh,
                              validation_data=(X_val_p, y_val_oh),
                              epochs=200, batch_size=16, verbose=0)

# 3b. Regularized: dropout + L2 + early stopping
reg_model = build_model(chosen_arch, dropout=0.3, l2=1e-3)
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True)
reg_hist = reg_model.fit(X_train_p, y_train_oh,
                          validation_data=(X_val_p, y_val_oh),
                          epochs=200, batch_size=16,
                          callbacks=[early_stop], verbose=0)

# Plot loss curves: plain vs regularized
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(plain_hist.history['loss'], label='train loss')
axes[0].plot(plain_hist.history['val_loss'], label='val loss')
axes[0].set_title('No regularization')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend()

axes[1].plot(reg_hist.history['loss'], label='train loss')
axes[1].plot(reg_hist.history['val_loss'], label='val loss')
axes[1].set_title('Dropout + L2 + Early Stopping')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend()
plt.tight_layout()
plt.savefig('overfitting_comparison.png', dpi=150)
plt.close()

plain_train_acc = plain_hist.history['accuracy'][-1]
plain_val_acc = plain_hist.history['val_accuracy'][-1]
reg_train_acc = reg_hist.history['accuracy'][-1]
reg_val_acc = reg_hist.history['val_accuracy'][-1]

print(f"\nPlain model:      final train_acc={plain_train_acc:.3f}, val_acc={plain_val_acc:.3f}, gap={plain_train_acc-plain_val_acc:.3f}")
print(f"Regularized model: final train_acc={reg_train_acc:.3f}, val_acc={reg_val_acc:.3f}, gap={reg_train_acc-reg_val_acc:.3f}")
print(f"Regularized model stopped at epoch {len(reg_hist.history['loss'])} (early stopping)")


# 4. Final evaluation 

test_loss, test_acc = reg_model.evaluate(X_test_p, y_test_oh, verbose=0)
print(f"\nTest accuracy (final model): {test_acc:.3f}")

y_pred_prob = reg_model.predict(X_test_p, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

report = classification_report(y_test, y_pred, target_names=class_names)
print(report)
with open('classification_report.txt', 'w') as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

# Save final model + preprocessing objects
reg_model.save('recruitment_model.keras')

import joblib
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(le, 'label_encoder.joblib')

# Save summary metrics for report generation
import json
summary = {
    'architecture_results': results,
    'plain_final': {'train_acc': float(plain_train_acc), 'val_acc': float(plain_val_acc)},
    'reg_final': {'train_acc': float(reg_train_acc), 'val_acc': float(reg_val_acc)},
    'reg_epochs_trained': len(reg_hist.history['loss']),
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
    'class_names': class_names.tolist(),
    'train_size': len(X_train), 'val_size': len(X_val), 'test_size': len(X_test),
    'input_dim': int(input_dim),
}
with open('summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nDone. Model and artifacts saved.")
