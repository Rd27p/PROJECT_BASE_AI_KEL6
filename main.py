import math
import csv

# ===================== CSV Reader =====================
def read_csv(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip().split(',')
        data = []
        for line in lines[1:]:
            parts = line.strip().split(',')
            features = list(map(float, parts[:-1]))
            label_str = parts[-1].lower()
            if label_str in ['positive', '1']:
                label = 1
            elif label_str in ['negative', '0']:
                label = 0
            else:
                raise ValueError(f"Label tidak dikenali: {label_str}")
            data.append(features + [label])
        return data, header
    
# ===================== Naive Bayes Core =====================
def separate_by_class(data):
    separated = {}
    for row in data:
        label = int(row[-1])
        separated.setdefault(label, []).append(row[:-1])
    return separated

def summarize_dataset(dataset):
    summaries = []
    for column in zip(*dataset):
        mean = sum(column) / len(column)
        variance = sum((x - mean) ** 2 for x in column) / len(column)
        summaries.append((mean, variance))
    return summaries

def gaussian_probability(x, mean, var):
    var = max(var, 1e-9)  # Hindari pembagian nol
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / math.sqrt(2 * math.pi * var)) * exponent

def calculate_class_probabilities(summaries, separated_train, row):
    total_rows = sum(len(rows) for rows in separated_train.values())
    class_priors = {cls: len(rows) / total_rows for cls, rows in separated_train.items()}

    probabilities = {}
    for class_value, class_summaries in summaries.items():
        prob = class_priors[class_value]
        for i, (mean, var) in enumerate(class_summaries):
            prob *= gaussian_probability(row[i], mean, var)
        probabilities[class_value] = prob
    return probabilities

def predict(summaries, separated_train, row):
    probabilities = calculate_class_probabilities(summaries, separated_train, row)
    return max(probabilities, key=probabilities.get)

# ===================== Evaluasi =====================
def confusion_elements(y_true, y_pred):
    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1: tp += 1
        elif yt == 0 and yp == 0: tn += 1
        elif yt == 0 and yp == 1: fp += 1
        elif yt == 1 and yp == 0: fn += 1
    return tp, tn, fp, fn

def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_elements(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return accuracy, specificity, precision, f1, tp, tn, fp, fn

# ===================== Leave-One-Out Cross Validation =====================
def loo_cross_validation(data):
    y_true_all, y_pred_all = [], []
    for i in range(len(data)):
        train = data[:i] + data[i+1:]
        test = data[i]
        separated = separate_by_class(train)
        summaries = {cls: summarize_dataset(rows) for cls, rows in separated.items()}
        pred = predict(summaries, separated, test[:-1])
        y_true_all.append(int(test[-1]))
        y_pred_all.append(pred)
    return y_true_all, y_pred_all

# ===================== Input Data Baru =====================
def input_user_data(headers):
    print("\nMasukkan data baru untuk prediksi:")
    input_values = []
    for feature in headers[:-1]:
        while True:
            try:
                val = float(input(f"{feature}: "))
                input_values.append(val)
                break
            except ValueError:
                print("Input harus berupa angka.")
    return input_values

# ===================== Evaluasi dan Cetak =====================
def evaluate_and_print(name, data, summaries, separated_train):
    y_true = [int(row[-1]) for row in data]
    y_pred = [predict(summaries, separated_train, row[:-1]) for row in data]
    acc, spec, prec, f1, tp, tn, fp, fn = compute_metrics(y_true, y_pred)
    print(f"\n=== Evaluasi Data {name} ===")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# ===================== Main =====================
def main():
    train_data, headers = read_csv('train_data.csv')
    val_data, _ = read_csv('val_data.csv')
    test_data, _ = read_csv('test_data.csv')

    # Train model
    separated_train = separate_by_class(train_data)
    summaries = {cls: summarize_dataset(rows) for cls, rows in separated_train.items()}

    # Evaluasi data
    evaluate_and_print("Train", train_data, summaries, separated_train)
    evaluate_and_print("Test", test_data, summaries, separated_train)

    # Evaluasi LOO-CV pada val_data
    print("\n=== Evaluasi Validasi (LOO-CV pada Validation Set) ===")
    y_true_val, y_pred_val = loo_cross_validation(val_data)
    acc, spec, prec, f1, tp, tn, fp, fn = compute_metrics(y_true_val, y_pred_val)
    print(f"Accuracy    : {acc:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # Prediksi input user
    new_data = input_user_data(headers)
    prediction = predict(summaries, separated_train, new_data)
    print(f"\n📊 Prediksi untuk data baru: task_success = {prediction} ({'BERHASIL' if prediction == 1 else 'GAGAL'})")

    print("\n=== Program Selesai ===")

if __name__ == '__main__':
    main()
