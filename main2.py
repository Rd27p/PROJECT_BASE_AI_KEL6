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

# ===================== KNN Core =====================
def euclidean_distance(row1, row2):
    # Hitung jarak Euclidean antar dua data (tanpa label)
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

def get_neighbors(train_data, test_row, k=3):
    distances = []
    for train_row in train_data:
        dist = euclidean_distance(test_row, train_row[:-1])
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

def predict(train_data, test_row, k=3):
    neighbors = get_neighbors(train_data, test_row, k)
    classes = [int(row[-1]) for row in neighbors]
    # Voting mayoritas
    prediction = max(set(classes), key=classes.count)
    return prediction

def get_predictions(train_data, test_data, k=3):
    predictions = []
    for row in test_data:
        output = predict(train_data, row[:-1], k)
        predictions.append(output)
    return predictions

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

    k = 3  # bisa diubah sesuai kebutuhan

    # Prediksi data latih dan uji
    y_train_true = [int(row[-1]) for row in train_data]
    y_test_true = [int(row[-1]) for row in test_data]
    y_train_pred = get_predictions(train_data, train_data, k)
    y_test_pred = get_predictions(train_data, test_data, k)

    train_metrics = compute_metrics(y_train_true, y_train_pred)
    test_metrics = compute_metrics(y_test_true, y_test_pred)

    # Hitung confusion matrix untuk data latih
    tp_train, tn_train, fp_train, fn_train = confusion_elements(y_train_true, y_train_pred)

    # Hitung confusion matrix untuk data uji
    tp_test, tn_test, fp_test, fn_test = confusion_elements(y_test_true, y_test_pred)

    # Tampilkan hasil evaluasi data latih
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
    prediction = predict(train_data, new_data, k)
    print(f"\n📊 Prediksi untuk data baru: task_success = {prediction} ({'BERHASIL' if prediction == 1 else 'GAGAL'})")

if __name__ == '__main__':
    main()
