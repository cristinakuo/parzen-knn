def get_error(label_real,label_test):
    label_error = [a - b for a, b in zip(label_real, label_test)]
    n_errors = len([n for n in label_error if n != 0])
    error_rate = n_errors / len(label_error)
    return error_rate
