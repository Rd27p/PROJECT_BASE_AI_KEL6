import csv
import sys

# === Encoding ===
def encode_gender(gender):
    return 1 if gender.strip().lower() == 'male' else 0

def encode_admission_type(admission_type):
    types = {'urgent': 0, 'emergency': 1, 'elective': 2}
    return types.get(admission_type.strip().lower(), -1)

# === Matrix Operations ===
def transpose(matrix):
    return list(map(list, zip(*matrix)))

def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def inverse_3x3(m):
    det = (
        m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
        - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
        + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0])
    )
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    inv = [[0]*3 for _ in range(3)]

    inv[0][0] =  (m[1][1]*m[2][2] - m[1][2]*m[2][1]) / det
    inv[0][1] = -(m[0][1]*m[2][2] - m[0][2]*m[2][1]) / det
    inv[0][2] =  (m[0][1]*m[1][2] - m[0][2]*m[1][1]) / det

    inv[1][0] = -(m[1][0]*m[2][2] - m[1][2]*m[2][0]) / det
    inv[1][1] =  (m[0][0]*m[2][2] - m[0][2]*m[2][0]) / det
    inv[1][2] = -(m[0][0]*m[1][2] - m[0][2]*m[1][0]) / det

    inv[2][0] =  (m[1][0]*m[2][1] - m[1][1]*m[2][0]) / det
    inv[2][1] = -(m[0][0]*m[2][1] - m[0][1]*m[2][0]) / det
    inv[2][2] =  (m[0][0]*m[1][1] - m[0][1]*m[1][0]) / det

    return inv

# === Load Data ===
def load_data(filepath):
    X, y, names, rows = [], [], [], []
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                age = float(row['Age'])
                gender = encode_gender(row['Gender'])
                admission = encode_admission_type(row['Admission Type'])
                billing = float(row['Billing Amount'])
                if admission == -1:
                    continue
                X.append([1.0, age, gender, admission])
                y.append(billing)
                names.append(row['Name'])
                rows.append(row)
            except:
                continue
    return X, y, names, rows

# === Linear Regression ===
def linear_regression_fit(X, y):
    X_T = transpose(X)
    XTX = matmul(X_T, X)
    XTy = matmul(X_T, [[val] for val in y])
    XTX_inv = inverse_3x3(XTX)
    theta = matmul(XTX_inv, XTy)
    return [t[0] for t in theta]

def predict(X, theta):
    return [sum(x_i * t_i for x_i, t_i in zip(x, theta)) for x in X]

# === Evaluation ===
def evaluate_classification(y_true, y_pred, threshold):
    TP = TN = FP = FN = 0
    for actual, pred in zip(y_true, y_pred):
        actual_class = 1 if actual > threshold else 0
        predicted_class = 1 if pred > threshold else 0
        if predicted_class == 1 and actual_class == 1: TP += 1
        elif predicted_class == 0 and actual_class == 0: TN += 1
        elif predicted_class == 1 and actual_class == 0: FP += 1
        elif predicted_class == 0 and actual_class == 1: FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return TP, TN, FP, FN, accuracy, precision, recall, f1

# === Fungsi Tambahan untuk Median dan Distribusi ===
def hitung_median(data):
    data_sorted = sorted(data)
    n = len(data_sorted)
    if n % 2 == 1:
        return data_sorted[n // 2]
    else:
        return (data_sorted[n // 2 - 1] + data_sorted[n // 2]) / 2

threshold_pred = hitung_median(predictions)
print(f"Threshold berdasarkan median prediksi: {threshold_pred:.2f}")

def print_distribusi_label(y_true, y_pred, threshold):
    count_pos = sum(1 for v in y_true if v > threshold)
    count_neg = sum(1 for v in y_true if v <= threshold)
    pred_pos = sum(1 for v in y_pred if v > threshold)
    pred_neg = sum(1 for v in y_pred if v <= threshold)
    print(f"Distribusi Label Aktual > {threshold}: {count_pos}")
    print(f"Distribusi Label Aktual <= {threshold}: {count_neg}")
    print(f"Distribusi Prediksi > {threshold}: {pred_pos}")
    print(f"Distribusi Prediksi <= {threshold}: {pred_neg}")

# === Main ===
def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_billing_with_metrics.py <train_csv> <test_csv>")
        return

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    threshold = 25000.0

    # Load training data
    X_train, y_train, _, _ = load_data(train_file)

    # Train model
    theta = linear_regression_fit(X_train, y_train)

    # Load test data
    X_test, y_test, names_test, rows_test = load_data(test_file)

    # Predict
    predictions = predict(X_test, theta)

    print(f"\n[INFO] Data uji: {len(y_test)} baris")
    print(f"[INFO] Prediksi minimum: {min(predictions):.2f}")
    print(f"[INFO] Prediksi maksimum: {max(predictions):.2f}")
    print(f"[INFO] Prediksi di atas threshold {threshold}: {sum(1 for p in predictions if p > threshold)}")

    print("\n=== Distribusi Label dan Prediksi dengan threshold default ===")
    print_distribusi_label(y_test, predictions, threshold)

    # Cek apakah ada label negatif, kalau tidak coba threshold median
    count_neg = sum(1 for v in y_test if v <= threshold)
    if count_neg == 0:
        median_threshold = hitung_median(y_test)
        print(f"\nTidak ada label <= {threshold}, coba threshold alternatif median: {median_threshold:.2f}")
        print_distribusi_label(y_test, predictions, median_threshold)
        threshold = median_threshold  # pakai threshold baru ini

    # Evaluasi dengan threshold yang sudah diperbaiki
    TP, TN, FP, FN, acc, prec, rec, f1 = evaluate_classification(y_test, predictions, threshold)

    print("\nConfusion Matrix dan Metrik:")
    print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
    print(f"Akurasi: {acc:.2f}")
    print(f"Presisi: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Save result
    output_file = "billing_predictions_evaluated.csv"
    with open(output_file, 'w', newline='') as file:
        fieldnames = list(rows_test[0].keys()) + ['Predicted Billing', 'Actual Label', 'Predicted Label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, actual, pred in zip(rows_test, y_test, predictions):
            row['Predicted Billing'] = round(pred, 2)
            row['Actual Label'] = 1 if actual > threshold else 0
            row['Predicted Label'] = 1 if pred > threshold else 0
            writer.writerow(row)

    print(f"\n[INFO] Hasil prediksi dan evaluasi disimpan di {output_file}")

if __name__ == "__main__":
    main()
