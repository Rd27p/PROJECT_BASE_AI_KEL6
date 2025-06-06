import math

# ===================== CSV Reader =====================
def read_csv(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip().split(',')
        data = []
        for line in lines[1:]:
            row = list(map(float, line.strip().split(',')))
            data.append(row)
        return data, header

# ===================== Naive Bayes Core =====================
def separate_by_class(data):
    separated = {}
    for row in data:
        label = int(row[-1])
        if label not in separated:
            separated[label] = []
        separated[label].append(row[:-1])
    return separated

def summarize_dataset(dataset):
    summaries = []
    for column in zip(*dataset):
        mean = sum(column) / len(column)
        variance = sum((x - mean) ** 2 for x in column) / len(column)
        summaries.append((mean, variance))
    return summaries

def gaussian_probability(x, mean, var):
    var = max(var, 1e-9)  # smoothing supaya variance tidak nol
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / math.sqrt(2 * math.pi * var)) * exponent

def calculate_class_probabilities(summaries, separated_train, row):
    total_rows = sum(len(rows) for rows in separated_train.values())
    class_priors = {cls: len(rows) / total_rows for cls, rows in separated_train.items()}

    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = class_priors[class_value]
        for i in range(len(class_summaries)):
            mean, var = class_summaries[i]
            probabilities[class_value] *= gaussian_probability(row[i], mean, var)
    return probabilities

def predict(summaries, separated_train, row):
    probabilities = calculate_class_probabilities(summaries, separated_train, row)
    return max(probabilities, key=probabilities.get)

def get_predictions(summaries, separated_train, test_data):
    return [predict(summaries, separated_train, row[:-1]) for row in test_data]

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
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return accuracy, specificity, precision, f1

# ===================== Input Data Baru =====================
def input_user_data(headers):
    print("\nMasukkan data baru untuk prediksi:")
    input_values = []
    for feature in headers[:-1]:  # exclude label
        while True:
            try:
                val = float(input(f"{feature}: "))
                input_values.append(val)
                break
            except ValueError:
                print("Input harus berupa angka.")
    return input_values

# ===================== Main Program =====================
def main():
    train_data, headers = read_csv('train_data.csv')
    test_data, _ = read_csv('test_data.csv')

    # Training
    separated_train = separate_by_class(train_data)
    summaries = {cls: summarize_dataset(rows) for cls, rows in separated_train.items()}

    # Evaluasi data latih dan uji
    y_train_true = [int(row[-1]) for row in train_data]
    y_test_true = [int(row[-1]) for row in test_data]
    y_train_pred = get_predictions(summaries, separated_train, train_data)
    y_test_pred = get_predictions(summaries, separated_train, test_data)

    train_metrics = compute_metrics(y_train_true, y_train_pred)
    test_metrics = compute_metrics(y_test_true, y_test_pred)

    # Hitung confusion matrix untuk data latih
    tp_train, tn_train, fp_train, fn_train = confusion_elements(y_train_true, y_train_pred)

    # Hitung confusion matrix untuk data uji
    tp_test, tn_test, fp_test, fn_test = confusion_elements(y_test_true, y_test_pred)

    # Tampilkan hasil evaluasi data latih dengan confusion matrix
    print("=== Evaluasi Data Latih ===")
    print(f"Accuracy    : {train_metrics[0]:.4f}")
    print(f"Specificity : {train_metrics[1]:.4f}")
    print(f"Precision   : {train_metrics[2]:.4f}")
    print(f"F1-Score    : {train_metrics[3]:.4f}")
    print(f"TP: {tp_train}, TN: {tn_train}, FP: {fp_train}, FN: {fn_train}")

    print("\n=== Evaluasi Data Uji ===")
    print(f"Accuracy    : {test_metrics[0]:.4f}")
    print(f"Specificity : {test_metrics[1]:.4f}")
    print(f"Precision   : {test_metrics[2]:.4f}")
    print(f"F1-Score    : {test_metrics[3]:.4f}")
    print(f"TP: {tp_test}, TN: {tn_test}, FP: {fp_test}, FN: {fn_test}")

    # Input data baru dari user
    new_data = input_user_data(headers)
    prediction = predict(summaries, separated_train, new_data)
    print(f"\n📊 Prediksi untuk data baru: task_success = {prediction} ({'BERHASIL' if prediction == 1 else 'GAGAL'})")

    # Hitung ulang prediksi dan metrik data uji tanpa menggunakan test_metrics lama
    y_test_pred_new = get_predictions(summaries, separated_train, test_data)
    test_metrics_new = compute_metrics(y_test_true, y_test_pred_new)

    print("\n=== Evaluasi Model pada Data Uji (Setelah Prediksi) ===")
    print(f"Accuracy    : {test_metrics_new[0]:.4f}")
    print(f"Specificity : {test_metrics_new[1]:.4f}")
    print(f"Precision   : {test_metrics_new[2]:.4f}")
    print(f"F1-Score    : {test_metrics_new[3]:.4f}")

if __name__ == '__main__':
    main()
