import os 
from glob import glob 
from sklearn.model_selection import train_test_split
import shutil

for VIEW in ['AP', 'LA']:
    DATASET_NAME = "BUU_LSPINE_V2_" + VIEW
    FOLDER_PATH = os.path.join('dataset','BUU_LSPINE_V2',VIEW)

    paths = glob(os.path.join(FOLDER_PATH, '*.jpg'))
    random_num = 42
    train_test_eval_ratio = [0.6, 0.3, 0.1]

    path_data = {
        'train': [],
        'test': [],
        'eval': []
    }
    path_data['train'], data_temp = train_test_split(paths, test_size=train_test_eval_ratio[1], random_state=random_num)
    path_data['test'], path_data['eval'] = train_test_split(data_temp, test_size=(train_test_eval_ratio[1]+train_test_eval_ratio[2])/2, random_state=random_num)

    OUTPUT_PATH = {
        'train':os.path.join('data', DATASET_NAME, 'train'),
        'test':os.path.join('data', DATASET_NAME, 'test'),
        'eval':os.path.join('data', DATASET_NAME, 'eval')
    }

    for path in OUTPUT_PATH.values():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    for path in OUTPUT_PATH.values():
        image_path = os.path.join(path, 'images')
        label_path = os.path.join(path, 'labels')
        if not os.path.exists(image_path):
            os.makedirs(image_path, exist_ok=True)
        if not os.path.exists(label_path):
            os.makedirs(label_path, exist_ok=True)

    for type in path_data:
        for path in path_data[type]:
            shutil.copy(path, os.path.join(OUTPUT_PATH[type], 'images'))
            shutil.copy(path.replace('.jpg','.csv'), os.path.join(OUTPUT_PATH[type], 'labels'))
        print(VIEW, type, 'done.')




