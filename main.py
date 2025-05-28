import csv
import sys

# === Encoding ===
def encode_gender(gender):
    return 1 if gender.strip().lower() == 'male' else 0

def encode_admission_type(admission_type):
    types = {'urgent': 0, 'emergency': 1, 'elective': 2}
    return types.get(admission_type.strip().lower(), -1)

# === Matrix Ops ===
def transpose(matrix):
    return list(map(list, zip(*matrix)))

def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def inverse_3x3(m):
    from copy import deepcopy
    n = len(m)
    I = [[float(i == j) for i in range(n)] for j in range(n)]
    A = deepcopy(m)
    for i in range(n):
        pivot = A[i][i]
        for j in range(n):
            A[i][j] /= pivot
            I[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(n):
                    A[k][j] -= factor * A[i][j]
                    I[k][j] -= factor * I[i][j]
    return I

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

# === Evaluation Metrics ===
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

# === Main CLI ===
def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_billing_with_metrics.py <csv_file_path>")
        return

    filepath = sys.argv[1]
    threshold = 30000.0
    X, y, names, rows = load_data(filepath)
    theta = linear_regression_fit(X, y)
    predictions = predict(X, theta)

    # Evaluation
    TP, TN, FP, FN, acc, prec, rec, f1 = evaluate_classification(y, predictions, threshold) 

    # Print
    print("Predicted Billing Amounts:")
    for name, pred in zip(names, predictions):
        print(f"{name}: ${pred:.2f}")

    print("\nConfusion Matrix and Metrics:")
    print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Write to CSV
    output_file = "billing_predictions_with_metrics.csv"
    with open(output_file, 'w', newline='') as file:
        fieldnames = list(rows[0].keys()) + ['Predicted Billing', 'Actual Label', 'Predicted Label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, actual, pred in zip(rows, y, predictions):
            row['Predicted Billing'] = round(pred, 2)
            row['Actual Label'] = 1 if actual > threshold else 0
            row['Predicted Label'] = 1 if pred > threshold else 0
            writer.writerow(row)

        # Write metrics at end
        writer.writerow({})
        writer.writerow({'Predicted Billing': 'TP', 'Actual Label': TP})
        writer.writerow({'Predicted Billing': 'TN', 'Actual Label': TN})
        writer.writerow({'Predicted Billing': 'FP', 'Actual Label': FP})
        writer.writerow({'Predicted Billing': 'FN', 'Actual Label': FN})
        writer.writerow({'Predicted Billing': 'Accuracy', 'Actual Label': f"{acc:.2f}"})
        writer.writerow({'Predicted Billing': 'Precision', 'Actual Label': f"{prec:.2f}"})
        writer.writerow({'Predicted Billing': 'Recall', 'Actual Label': f"{rec:.2f}"})
        writer.writerow({'Predicted Billing': 'F1 Score', 'Actual Label': f"{f1:.2f}"})

    print(f"\nAll results saved to {output_file}")

if __name__ == "__main__":
    main()
