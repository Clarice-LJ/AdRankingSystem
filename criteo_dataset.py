import kagglehub as kh

path = kh.dataset_download("mrkmakr/criteo-dataset")
print("Path to dataset:", path)

# Display the first few lines of the training dataset
train_file = path + "/dac/train.txt"
print("\nFirst 5 lines of the training dataset:")
with open(train_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"Line {i+1}: {line.strip()}")