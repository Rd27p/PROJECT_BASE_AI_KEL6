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
    var = max(var, 1e-9)
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

# ===================== Evaluasi =====================
def confusion_elements(y_true, y_pred):
    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return tp, tn, fp, fn

def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_elements(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return accuracy, specificity, precision, f1

# ===================== Leave-One-Out Cross Validation =====================
def loo_cross_validation(data):
    n = len(data)
    y_true_all = []
    y_pred_all = []
    for i in range(n):
        train = data[:i] + data[i+1:]
        test = data[i]

        separated_train = separate_by_class(train)
        summaries = {cls: summarize_dataset(rows) for cls, rows in separated_train.items()}
        prediction = predict(summaries, separated_train, test[:-1])

        y_true_all.append(int(test[-1]))
        y_pred_all.append(prediction)
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

# ===================== Main =====================
def main():
    train_data, headers = read_csv('train_data.csv')
    val_data, _ = read_csv('val_data.csv')
    test_data, _ = read_csv('test_data.csv')

    # === EVALUASI DATA TRAIN ===
    print("=== Evaluasi Data Train ===")
    separated_train = separate_by_class(train_data)
    summaries = {cls: summarize_dataset(rows) for cls, rows in separated_train.items()}
    y_true_train = [int(row[-1]) for row in train_data]
    y_pred_train = [predict(summaries, separated_train, row[:-1]) for row in train_data]
    acc, spec, prec, f1 = compute_metrics(y_true_train, y_pred_train)
    tp, tn, fp, fn = confusion_elements(y_true_train, y_pred_train)
    print(f"Accuracy    : {acc:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # === TESTING ===
    print("\n=== Evaluasi Data Uji ===")
    separated_train = separate_by_class(train_data)
    summaries = {cls: summarize_dataset(rows) for cls, rows in separated_train.items()}
    y_true_test = [int(row[-1]) for row in test_data]
    y_pred_test = [predict(summaries, separated_train, row[:-1]) for row in test_data]
    acc, spec, prec, f1 = compute_metrics(y_true_test, y_pred_test)
    tp, tn, fp, fn = confusion_elements(y_true_test, y_pred_test)
    print(f"Accuracy    : {acc:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # === VALIDASI (LOO-CV PADA VAL DATA) ===
    print("=== Evaluasi Validasi (LOO-CV) ===")
    y_true_val, y_pred_val = loo_cross_validation(val_data)
    acc, spec, prec, f1 = compute_metrics(y_true_val, y_pred_val)
    tp, tn, fp, fn = confusion_elements(y_true_val, y_pred_val)
    print(f"Accuracy    : {acc:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # === PREDIKSI DATA BARU ===
    new_data = input_user_data(headers)
    prediction = predict(summaries, separated_train, new_data)
    print(f"\n📊 Prediksi untuk data baru: task_success = {prediction} ({'BERHASIL' if prediction == 1 else 'GAGAL'})")
    print("\n=== Program Selesai ===")
    
if __name__ == '__main__':
    main()
