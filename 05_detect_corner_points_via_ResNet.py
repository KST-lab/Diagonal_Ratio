from tensorflow.keras.models import load_model
import os
from glob import glob
import cv2
import numpy as np
import json
import pandas as pd

def euclidean_distance(point1, point2):
    # Convert the points to numpy arrays for easy mathematical operations
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the distance
    distance = np.sqrt(np.sum((point1 - point2)**2))

    return distance

resize_size = (250, 250, 3)
json_output = {
    'L1':[],
    'L2':[],
    'L3':[],
    'L4':[],
    'L5':[],
}

# ---- get PIXEL SPACING (to convert unit from pixels to Millimetre)----
PIXEL_SPACING_PATH = os.path.join('csv','pixel_spacing_BUU_LSPINE_V2.csv')
df_pixel_spacing = pd.read_csv(PIXEL_SPACING_PATH,sep='\t', index_col='filename')
df_pixel_spacing_dict = df_pixel_spacing.to_dict(orient='index')

def pixel_spacing_mapping(filename_idx):
    return df_pixel_spacing_dict[filename_idx]['pixel_spacing']
# ----------------------------------------------------------------------


for data_set in ['test']:
    for VIEW in ['AP', 'LA']:
        MODEL_NAME_ARRAY = [VIEW + '_ResNet50V2.h5', VIEW + '_ResNet101V2.h5', VIEW + '_ResNet152V2.h5']
        for MODEL_NAME in MODEL_NAME_ARRAY:
            MAE_res = []
            # define folder to save detected corner points
            DETECT_POINTS_FOLDER = os.path.join('data', 'BUU_LSPINE_V2_' + VIEW, data_set, 'detect_points', MODEL_NAME)
            if not os.path.exists(DETECT_POINTS_FOLDER):
                os.makedirs(DETECT_POINTS_FOLDER, exist_ok=True)

            # load model
            MODEL_PATH = os.path.join('models', MODEL_NAME)
            model = load_model(MODEL_PATH)

            # iterate images in test set
            IMAGES_PATH = os.path.join('data', 'BUU_LSPINE_V2_' + VIEW, data_set, 'images')
            for IMAGE_PATH in glob(IMAGES_PATH + '/*.jpg'):
                # get filename
                filename = os.path.basename(IMAGE_PATH)

                # get pixel spacing
                filename_idx = filename[0:-5]
                pixel_spacing = df_pixel_spacing_dict[filename_idx]['pixel_spacing']
                
                # load ground truth points 
                LABEL_PATH = IMAGE_PATH.replace('images', 'labels').replace('.jpg', '.json')
                with open(LABEL_PATH, 'r', encoding = "utf-8") as f:
                    label = json.load(f)
                ground_truth_corner_points = [label['keypoints']]

                # load image
                img = cv2.imread(IMAGE_PATH)
                h, w, d = img.shape

                # resize image and normalize intensity
                resized_img = cv2.resize(img, (250,250))
                resized_img = np.array([resized_img/255])

                # predict corner points via ResNet
                detect_corner_points_via_ResNet = model.predict(resized_img)[0]

                # reshape points array 
                # print(len(detect_corner_points_via_ResNet[0]))
                # print(detect_corner_points_via_ResNet)
                # quit()
                number_of_points = int(len(detect_corner_points_via_ResNet)//2) # 20 points for AP, 22 points for LA
                predicted_corner_points = np.array(detect_corner_points_via_ResNet).reshape((number_of_points, 2)) * (w, h)
                predicted_corner_points = np.array(predicted_corner_points, dtype=np.uint32)
                ground_truth_corner_points = np.array(ground_truth_corner_points).reshape((number_of_points, 2)) * (w, h)
                ground_truth_corner_points = np.array(ground_truth_corner_points, dtype=np.uint32)

                # reshape points for export in json files
                export_predicted_corner_points = predicted_corner_points.reshape(number_of_points//2,2,2)

                # save detected corner points via ResNet to json files.
                json_output['L1'] = export_predicted_corner_points[0].tolist() + export_predicted_corner_points[1].tolist()
                json_output['L2'] = export_predicted_corner_points[2].tolist() + export_predicted_corner_points[3].tolist()
                json_output['L3'] = export_predicted_corner_points[4].tolist() + export_predicted_corner_points[5].tolist()
                json_output['L4'] = export_predicted_corner_points[6].tolist() + export_predicted_corner_points[7].tolist()
                json_output['L5'] = export_predicted_corner_points[8].tolist() + export_predicted_corner_points[9].tolist()
                if VIEW == 'LA':
                    json_output['S1a'] = export_predicted_corner_points[10].tolist()
                json_file_path = os.path.join(DETECT_POINTS_FOLDER, filename[0:-4]+'.json')
                with open(json_file_path, "w") as json_file:
                    json.dump(json_output, json_file)
                # print(json_output)

                # calculate MAE between detected corner points and ground truth points
                MAE_row = [filename]
                
                for i, points in enumerate(predicted_corner_points):
                    # print(points)
                    point1 = points
                    point2 = ground_truth_corner_points[i]
                    error = euclidean_distance(point1, point2) * pixel_spacing
                    # print(error)
                    MAE_row.append(error)
                MAE_res.append(MAE_row) 
                # break
            OUTPUT_CSV_RESNET_MAE_FOLDER = os.path.join('csv', 'ResNet')
            if not os.path.exists(OUTPUT_CSV_RESNET_MAE_FOLDER):
                os.makedirs(OUTPUT_CSV_RESNET_MAE_FOLDER, exist_ok=True)
            OUTPUT_CSV_RESNET_MAE_PATH = os.path.join(OUTPUT_CSV_RESNET_MAE_FOLDER, MODEL_NAME[0:-3] + '.csv')
            MAE_df = pd.DataFrame(MAE_res)
            MAE_df.to_csv(OUTPUT_CSV_RESNET_MAE_PATH)
        #     print(MAE_df)
        # quit()


            


