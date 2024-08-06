import os
from glob import glob 
import pandas as pd
import json
import numpy as np

for VIEW in ['AP', 'LA']:
    DATASET_FOLDER_PATH = os.path.join('data', 'BUU_LSPINE_V2_' + VIEW)
    IMAGE_SIZE_CSV_PATH = os.path.join('csv', 'image_size_' + VIEW + '_BUU_LSPINE_V2.csv')

    # Get images size
    df_image_size = pd.read_csv(IMAGE_SIZE_CSV_PATH)
    df_image_size['filename'] = df_image_size['filename'].str[:-4]
    df_image_size = df_image_size.set_index('filename')
    df_image_size = df_image_size.to_dict(orient='index')

    # Convert
    lumbar_map = {
        0: 'L1',
        1: 'L2',
        2: 'L3',
        3: 'L4',
        4: 'L5'
    }
    l = 0
    for data_set in ['train', 'test', 'eval']:
        OUTPUT_PATH = os.path.join(DATASET_FOLDER_PATH, data_set, 'labels')
        LABEL_PATH = os.path.join(DATASET_FOLDER_PATH, data_set, 'labels')   
        COOR_PATH = os.path.join(DATASET_FOLDER_PATH, data_set, 'coors')   
        IMAGE_PATH = os.path.join(DATASET_FOLDER_PATH, data_set, 'images') 
        if not os.path.exists(COOR_PATH):
            os.makedirs(COOR_PATH, exist_ok=True)
        
        for path in glob(os.path.join(LABEL_PATH,'*.csv')):
            # ---- normalized coordinates for ResNet---
            filename = os.path.basename(path)[0:-4]
            df = pd.read_csv(path,header=None)
            image_height = df_image_size[filename]['height']
            image_width = df_image_size[filename]['width']
            # print(path,filename, [image_height, image_width])
            df[0] = df[0]/image_width
            df[2] = df[2]/image_width
            df[1] = df[1]/image_height
            df[3] = df[3]/image_height
            if VIEW == 'AP':
                coordinates = df.loc[:,0:3].to_numpy().reshape([20,2])
            elif VIEW == 'LA':
                coordinates = df.loc[:,0:3].to_numpy().reshape([22,2])

            k = 0
            output_json = {
                'image': filename + '.jpg',
                'class': list([]),
                'keypoints': list([])
            }
            for lumbar in lumbar_map:
                for i in ['a', 'b']:
                    for j in ['1', '2']:
                        classname = lumbar_map[lumbar] + '_' + i + '_' + j
                        # output_json['class'].append(classname)
                        output_json['class'].append(1)
                        output_json['keypoints'].append(list(coordinates[k]))
                        # print(filename + '.jpg', classname, coordinates[k])                    
                        k = k + 1

            if VIEW == 'LA':
                # output_json['class'].append('S1_a_1')
                output_json['class'].append(1)
                output_json['keypoints'].append(list(coordinates[k]))
                # output_json['class'].append('S1_a_2')
                output_json['class'].append(1)
                output_json['keypoints'].append(list(coordinates[k + 1]))
            
            output_json['keypoints'] = [val for sublist in output_json['keypoints'] for val in sublist]


            # print(output_json)
            with open(os.path.join(OUTPUT_PATH, filename + '.json'), 'w') as f:
                json.dump(output_json, f)
            # break
            if VIEW == 'AP':
                lumbar_coor = {
                    'L1':[],
                    'L2':[],
                    'L3':[],
                    'L4':[],
                    'L5':[]
                }
            elif VIEW == 'LA':
                lumbar_coor = {
                    'L1':[],
                    'L2':[],
                    'L3':[],
                    'L4':[],
                    'L5':[],
                    'S1a':[]
                }
            else:
                break
            filename = os.path.basename(path)[0:-4]
            coor_df = pd.read_csv(path, header=None)
            coor_np = coor_df[[0,1,2,3]].to_numpy()
            h, w = coor_np.shape
            all_elem = h * w 
            coor_np = np.reshape(coor_np,(all_elem//2, 2))
            for i in range(5):
                lumbar_coor[lumbar_map[i]] = np.array(np.rint(coor_np[4 * i : 4 * i + 4]), dtype=np.int16).tolist()
            if VIEW == 'LA':
                lumbar_coor['S1a'] = np.array(np.rint(coor_np[20 : 22]), dtype=np.int16).tolist()
            with open(COOR_PATH + '/' + filename + '.json', 'w') as f:
                json.dump(lumbar_coor, f)
        print('normalize for ', VIEW, data_set)
        l = l + 1

