from sklearn.model_selection import train_test_split, StratifiedKFold
import glob
from utils.util import read_json
from pathlib import Path

types = ['单柱顺转1-1', '双柱顺转1-1', '三柱顺转1-1', '四柱顺转1-1', 'T型顺转1-1']
x = []
y = []
root_dir = "../"
json_file = read_json("E:\G2\G2codev2\experiment_of_zmx\\presudo.json")
pseudo = True
for i, t in enumerate(types):
    pictures = glob.glob(root_dir + '/images_train_5s/' + t + '/*.jpg')
    x.extend(pictures)
    y.extend([i] * len(pictures))
print(len(x))
labels = {'single': 0, 'double': 1, 'triple': 2, 'quadruple': 3, 'T-shape': 4}
if pseudo:
    for i, img_key in enumerate(json_file):
        data_path = 'E:/G2/G2codev2/images_test_A/'
        label_key = json_file[img_key]
        label = labels[label_key]
        data_path = data_path + (img_key + '.jpg')
        x.extend([data_path])
        y.extend([label])

print(len(x))
print(len(y))
skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
fold = 2
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    if i != fold:
        continue
    # print(i,"TRAIN:", train_index, "TEST:", test_index)
    x_train = [x[i] for i in train_index]
    x_test = [x[i] for i in test_index]
    y_train = [y[i] for i in train_index]
    y_test = [y[i] for i in test_index]
print(x_train[200])
print(y_train[200])