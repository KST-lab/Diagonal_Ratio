import os
import pandas as pd
from glob import glob
import json
import math
import time
import numpy as np

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_center(point1, point2):
    # Calculate the center coordinates
    center_x = (point1[0] + point2[0]) / 2
    center_y = (point1[1] + point2[1]) / 2
    return [center_x, center_y]  # Return the center point as a tuple

def cal_shift_distance(p1, p2, p3):
    # Calculate vectors
    v1 = [p1[0] - p2[0], p1[1] - p2[1]]
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]
    
    # Calculate dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    # Calculate magnitudes
    # magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)

    shift_distance = dot_product / magnitude2
    return shift_distance

def calculate_angle_between_lines(line_1, line_2):
    v1 = convert_to_vector(line_1[0],line_1[1])
    v2 = convert_to_vector(line_2[0],line_2[1])
    angle_1 = np.angle(v1,deg=True)
    angle_2 = np.angle(v2,deg=True)
    angle = angle_1 - angle_2
    return angle

def convert_to_vector(p1 = [0, 0], p2 = [0, 0]):
    return complex(p2[0]-p1[0], p2[1]-p1[1])


# ---- get PIXEL SPACING (to convert unit from pixels to Millimetre)----
PIXEL_SPACING_PATH = os.path.join('csv','pixel_spacing_BUU_LSPINE_V2.csv')
df_pixel_spacing = pd.read_csv(PIXEL_SPACING_PATH,sep='\t', index_col='filename')
df_pixel_spacing_dict = df_pixel_spacing.to_dict(orient='index')

def pixel_spacing_mapping(filename_idx):
    return df_pixel_spacing_dict[filename_idx]['pixel_spacing']
# ----------------------------------------------------------------------


for points_from in ['gt', 'dt']: # gt = ground truth, dt = detected points by ResNet')
    # if points_from == 'dt':
    #     continue
    for view in ['AP', 'LA']:
        for data_set  in ['train', 'test', 'eval']:
            if points_from == 'gt':
                # ---- From Ground Truth Points ----
                COORS_PATH = os.path.join('data', 'BUU_LSPINE_V2_'+view, data_set, 'coors')
            else:
                # ---- From Detected Points ----
                COORS_PATH = os.path.join('data', 'BUU_LSPINE_V2_'+view, data_set, 'detect_points', view + '_ResNet152V2.h5')
            res = []
            diag_df = pd.read_csv('csv/diag_all_BUU_LSPINE_V2.csv')
            diag_df['index'] = (diag_df['filename'] + diag_df['lumbar']).str.replace('b', '')
            diag_df = diag_df.set_index('index')
            diag_map = diag_df.to_dict(orient='index')
            couple_lumbar_map = {
                'L1': 'L2',
                'L2': 'L3',
                'L3': 'L4',
                'L4': 'L5',
                'L5': 'S1a',
            }
            for path in glob(COORS_PATH + '/*.json'):
                filename = os.path.basename(path)[0:-5]
                gender = filename[5]
                age = int(filename[7:10])
                f = open(path)
                data = json.load(f)
                lumbar_list = ['L1', 'L2', 'L3', 'L4', 'L5']
                if view == 'LA':
                    lumbar_list = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1a']
                for lumbar in lumbar_list:
                    if ((view == 'LA' and lumbar == 'S1a') or view == 'AP' and lumbar == 'L5'):
                        continue
                    lower_lumbar = couple_lumbar_map[lumbar]
                    upper_b_center = find_center(data[lumbar][2], data[lumbar][3])
                    lower_a_center = find_center(data[lower_lumbar][0], data[lower_lumbar][1])

                    # --- Spondylolisthesis features calculate ---
                    # 1. DH 
                    t1 = time.time()
                    DH = distance(lower_a_center, upper_b_center)
                    t2 = time.time()
                    time_DH = t2 - t1

                    # 2. SA (slip angle)
                    t1 =  time.time()
                    line_1 = np.array([data[lumbar][2],data[lumbar][3]])
                    line_2 = np.array([data[lower_lumbar][0],data[lower_lumbar][1]])
                    SA = calculate_angle_between_lines(line_1, line_2)
                    t2 =  time.time()
                    time_SA = t2 - t1

                    # 3. AD (angular deviation metric)
                    t1 =  time.time()
                    upper_center_point = find_center(find_center(data[lumbar][0], data[lumbar][3]), find_center(data[lumbar][1], data[lumbar][2]))
                    try:
                        lower_center_point = find_center(find_center(data[lower_lumbar][0], data[lower_lumbar][3]), find_center(data[lower_lumbar][1], data[lower_lumbar][2]))
                    except:
                        lower_center_point = find_center(data[lower_lumbar][0], data[lower_lumbar][1])
                    line_1 = np.array([upper_center_point, lower_center_point])
                    line_2 = np.array([upper_center_point, np.array([upper_center_point[0], lower_center_point[1]])])
                    AD = calculate_angle_between_lines(line_1, line_2)
                    t2 =  time.time()
                    time_AD = t2 - t1
                    
                    # 4. PSD_L (Piecewise Slope Detection's angle on the left side)
                    upper_b_r_point = data[lumbar][3]
                    upper_b_l_point = data[lumbar][2]
                    lower_a_r_point = data[lower_lumbar][1]
                    lower_a_l_point = data[lower_lumbar][0]     
                    t1 =  time.time()   
                    line_1 = np.array([upper_b_l_point, lower_a_l_point])
                    line_2 = np.array([upper_b_l_point, np.array([upper_b_l_point[0], lower_a_l_point[1]])])
                    PSD_L = calculate_angle_between_lines(line_1, line_2) 
                    t2 =  time.time()
                    time_PSD_L = t2 - t1  

                    # 5. PSD_R (Piecewise Slope Detection's angle on the right side)
                    t1 =  time.time()
                    line_1 = np.array([upper_b_r_point, lower_a_r_point])
                    line_2 = np.array([upper_b_r_point, np.array([upper_b_r_point[0], lower_a_r_point[1]])])
                    PSD_R = calculate_angle_between_lines(line_1, line_2)
                    t2 =  time.time()
                    time_PSD_R = t2 - t1

                    # 6. SDL (Shift Distance on the left side)
                    t1 = time.time()                   
                    w_lower = distance(data[lower_lumbar][0], data[lower_lumbar][1])
                    SDL = cal_shift_distance(data[lumbar][2], data[lower_lumbar][0], data[lower_lumbar][1])/w_lower
                    t2 = time.time()       
                    time_SDL = t2 - t1 

                    # 7. SDR (Shift Distance on the right side)
                    t1 = time.time()                   
                    w_lower = distance(data[lower_lumbar][0], data[lower_lumbar][1])
                    SDR = - cal_shift_distance(data[lumbar][3], data[lower_lumbar][1], data[lower_lumbar][0])/w_lower
                    t2 = time.time()       
                    time_SDR = t2 - t1 

                    # 8. ASD (Average Shift Distance between SDL and SDR)
                    t1 = time.time()                   
                    w_lower = distance(data[lower_lumbar][0], data[lower_lumbar][1])
                    SDL = cal_shift_distance(data[lumbar][2], data[lower_lumbar][0], data[lower_lumbar][1])/w_lower
                    SDR = - cal_shift_distance(data[lumbar][3], data[lower_lumbar][1], data[lower_lumbar][0])/w_lower
                    ASD = (SDL + SDR) / 2
                    t2 = time.time()       
                    time_ASD = t2 - t1 

                    # 9. DR (Diagonal Ratio) (Proposed feature)
                    t1 = time.time()
                    diagonal_line_1 = distance(data[lumbar][2], data[lower_lumbar][1])
                    diagonal_line_2 = distance(data[lumbar][3], data[lower_lumbar][0])
                    w_lower = distance(data[lower_lumbar][0], data[lower_lumbar][1])
                    DR = (diagonal_line_2 - diagonal_line_1)/w_lower # proposed diagonal ratio features
                    t2 = time.time()
                    time_DR =  t2 - t1 

                    # diag_num (groud truth spondylolisthesis class): 
                    diag_index = filename + lumbar
                    diag_num = 0
                    try:
                        diag_num = diag_map[diag_index]['diagnosis_number']
                    except:
                        diag_num = 0
                    row = [filename, gender, age, lumbar,
                           DH, SA, AD, PSD_L, PSD_R,
                           SDL, SDR, ASD,
                           DR,
                           time_DH, time_SA, time_AD, time_PSD_L, time_PSD_R,
                           time_SDL, time_SDR, time_ASD,
                           time_DR,
                           diag_num
                    ]
                    res.append(row)

            # --- create DataFrame for spondylolisthesis features ---
            cols = ['filename', 'gender', 'age', 'lumbar', 'DH', 'SA', 'AD', 'PSD_L', 'PSD_R', 'SDL', 'SDR', 'ASD', 'DR', 
                    'time_DH', 'time_SA', 'time_AD', 'time_PSD_L', 'time_PSD_R', 'time_SDL', 'time_SDR', 'time_ASD', 'time_DR', 'diag_num']
            res_df = pd.DataFrame(res, columns=cols)

            # --- insert pixel spacing column ---
            res_df['filename_index'] = res_df['filename'].apply(lambda x: x[:-1])
            res_df['pixel_spacing'] = res_df['filename_index'].apply(pixel_spacing_mapping)

            # --- convert feature unit from pixels to millimeter ) ---
            res_df['DH'] = res_df['DH'] * res_df['pixel_spacing']

            # --- save spondylolisthesis features csv file ----
            SPONDY_FEATURE_CSV_FOLDER = os.path.join('csv', 'spondylolisthesis_features_' + points_from)
            CSV_NAME = 'spondy_features_' + points_from + '_' + view + '_' + data_set + '.csv'
            if not os.path.exists(SPONDY_FEATURE_CSV_FOLDER):
                os.makedirs(SPONDY_FEATURE_CSV_FOLDER, exist_ok=True)
            res_df.to_csv(os.path.join(SPONDY_FEATURE_CSV_FOLDER, CSV_NAME))
            print(os.path.join(SPONDY_FEATURE_CSV_FOLDER, CSV_NAME), 'saved.')