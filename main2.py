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

# ===================== Decision Tree Core =====================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

def gini_index(groups, classes):
    n_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in classes:
            count = labels.count(class_val)
            proportion = count / size
            score += proportion ** 2
        gini += (1.0 - score) * (size / n_instances)
    return gini

def split_data(index, threshold, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_threshold, best_score, best_groups = None, None, float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            threshold = row[index]
            groups = split_data(index, threshold, dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_threshold, best_score, best_groups = index, threshold, gini, groups
    return {'index': best_index, 'threshold': best_threshold, 'groups': best_groups}

def to_terminal(group):
    counts = {}
    for row in group:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)

def build_tree(dataset, max_depth, min_size, depth=1):
    split = get_best_split(dataset)
    if not split or depth >= max_depth:
        return Node(value=to_terminal(dataset))
    
    left, right = split['groups']
    if len(left) == 0 or len(right) == 0:
        return Node(value=to_terminal(left + right))

    node = Node(feature=split['index'], threshold=split['threshold'])
    node.left = build_tree(left, max_depth, min_size, depth + 1) if len(left) > min_size else Node(value=to_terminal(left))
    node.right = build_tree(right, max_depth, min_size, depth + 1) if len(right) > min_size else Node(value=to_terminal(right))
    return node

def predict_tree(node, row):
    if node.is_leaf():
        return node.value
    if row[node.feature] < node.threshold:
        return predict_tree(node.left, row)
    else:
        return predict_tree(node.right, row)

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
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return accuracy, specificity, precision, f1, tp, tn, fp, fn

# ===================== LOO-CV =====================
def loo_cross_validation(data, max_depth, min_size):
    y_true_all, y_pred_all = [], []
    for i in range(len(data)):
        train = data[:i] + data[i+1:]
        test = data[i]
        tree = build_tree(train, max_depth, min_size)
        pred = predict_tree(tree, test[:-1])
        y_true_all.append(test[-1])
        y_pred_all.append(pred)
    return y_true_all, y_pred_all

# ===================== Input Manual =====================
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

# ===================== Evaluasi & Tampilkan Salah =====================
def evaluate_and_print(name, data, tree, show_wrong=False):
    y_true = [row[-1] for row in data]
    y_pred = []
    wrong_data = None
    for row in data:
        pred = predict_tree(tree, row[:-1])
        y_pred.append(pred)
        if show_wrong and pred != row[-1] and not wrong_data:
            wrong_data = row
    acc, spec, prec, f1, tp, tn, fp, fn = compute_metrics(y_true, y_pred)
    print(f"\n=== Evaluasi Data {name} ===")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    if show_wrong and wrong_data:
        print("\nContoh data TEST yang salah prediksi:")
        print("Fitur :", wrong_data[:-1])
        print("Prediksi:", predict_tree(tree, wrong_data[:-1]))

# ===================== Main =====================
def main():
    max_depth = 5
    min_size = 1

    train_data, headers = read_csv('train_data.csv')
    val_data, _ = read_csv('val_data.csv')
    test_data, _ = read_csv('test_data.csv')

    tree = build_tree(train_data, max_depth, min_size)

    evaluate_and_print("Train", train_data, tree)
    evaluate_and_print("Test", test_data, tree, show_wrong=True)

    print("\n=== Evaluasi Validasi (LOO-CV pada Validation Set) ===")
    y_true_val, y_pred_val = loo_cross_validation(val_data, max_depth, min_size)
    acc, spec, prec, f1, tp, tn, fp, fn = compute_metrics(y_true_val, y_pred_val)
    print(f"Accuracy    : {acc:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    new_data = input_user_data(headers)
    prediction = predict_tree(tree, new_data)
    print(f"\nPrediksi untuk data baru: {prediction} ({'BERHASIL' if prediction == 1 else 'GAGAL'})")

    print("\n=== Program Selesai ===")

if __name__ == '__main__':
    main()
