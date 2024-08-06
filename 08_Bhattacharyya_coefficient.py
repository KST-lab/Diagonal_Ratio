import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def bhattacharyya_coefficient(P, Q):
    input_bins = np.linspace(-10, 10, num=100)
    hist_P, bins_P = np.histogram(P, bins=input_bins)
    hist_P_norm = hist_P/np.sum(hist_P)
    hist_Q, bins_Q = np.histogram(Q, bins=input_bins)
    hist_Q_norm = hist_Q/np.sum(hist_Q)
    bc = np.sum(np.sqrt(hist_P_norm * hist_Q_norm))
    return bc

def remove_outlier(df, feature, class_num):
    temp_data = df[df['diag_num'] == class_num][feature]
    Q1 = np.percentile(temp_data, 20)
    Q3 = np.percentile(temp_data, 80)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[((df['diag_num']==class_num) & ((df[feature] <= lower_bound) | (df[feature] >= upper_bound))) == False]
    return df

FIGURES_FOLDER = os.path.join('figures', 'bhat')
if not os.path.exists(FIGURES_FOLDER):
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    
CSV_FOLDER = os.path.join('csv', 'bhat')
if not os.path.exists(CSV_FOLDER):
    os.makedirs(CSV_FOLDER, exist_ok=True)

fig_height = 10
fig_width = 9

for points_from in ['gt', 'dt']: # gt = ground truth, dt = detected points by ResNet')
    if points_from == 'dt':
        continue
    for view in ['AP', 'LA']:
        print('Analyzing Bhattachcaryya coefficient of feature on ' + view + ' view ...')
        # --- load features data ---
        CSV_PATH = os.path.join('csv', 'spondylolisthesis_features_' + points_from, 'spondy_features_'+points_from+'_'+view+'_train.csv')
        df1 = pd.read_csv(CSV_PATH, index_col=0)
        CSV_PATH = os.path.join('csv', 'spondylolisthesis_features_' + points_from, 'spondy_features_'+points_from+'_'+view+'_test.csv')
        df2 = pd.read_csv(CSV_PATH, index_col=0)
        CSV_PATH = os.path.join('csv', 'spondylolisthesis_features_' + points_from, 'spondy_features_'+points_from+'_'+view+'_eval.csv')
        df3 = pd.read_csv(CSV_PATH, index_col=0)
        frames = [df1, df2, df3]
        spondy_feature_df = pd.concat(frames, ignore_index=True)

        # --- change class number to represent direction ---
        spondy_feature_df.loc[spondy_feature_df['diag_num'] == 1, 'diag_num'] = 1
        spondy_feature_df.loc[spondy_feature_df['diag_num'] == 2, 'diag_num'] = -1
        spondy_feature_df.loc[spondy_feature_df['diag_num'] == 3, 'diag_num'] = 1
        spondy_feature_df.loc[spondy_feature_df['diag_num'] == 4, 'diag_num'] = -1

        # --- analyze feature distributions ---
        features_array = ['DH', 'SA', 'AD', 'SDL', 'SDR', 'ASD', 'DR']
        output_df = pd.DataFrame()  
        bhat_res = []
        for feature in features_array:
            # normalize feature value
            df_normalized = pd.DataFrame()
            df_normalized['diag_num'] = spondy_feature_df['diag_num']
            df_normalized['time_' + feature] = spondy_feature_df['time_' + feature]
            df_normalized[feature] = (spondy_feature_df[feature] - spondy_feature_df[feature].mean()) / spondy_feature_df[feature].std()
            
            # measure Bhattacharyya coefficients 
            b1 = bhattacharyya_coefficient(df_normalized[df_normalized['diag_num']==0][feature], df_normalized[df_normalized['diag_num']==1][feature])
            b2 = bhattacharyya_coefficient(df_normalized[df_normalized['diag_num']==0][feature], df_normalized[df_normalized['diag_num']==-1][feature])

            # remove outlier
            df_normalized =  remove_outlier(df_normalized, feature, 0)

            # set information in dataframe to plot distribution
            string_list = [feature for i in range(0, len(df_normalized))]
            plot_df = pd.DataFrame(string_list,columns=['x'])
            plot_df = plot_df.assign(y=df_normalized[feature].to_list())
            plot_df = plot_df.assign(diag_num=df_normalized['diag_num'].to_list())
            output_df = pd.concat([output_df, plot_df],axis=0,ignore_index=True)

            # calculate computational time 
            computational_time = df_normalized['time_' + feature].mean() * 1000000 
            bhat_row = [feature, b1, b2, computational_time]
            bhat_res.append(bhat_row)

        # change class titles for legend in figure
        output_df['diag_class'] = output_df['diag_num']
        if view == 'AP':
            output_df.loc[output_df['diag_class'] == 0, 'diag_class'] = 'NORM'
            output_df.loc[output_df['diag_class'] == 1, 'diag_class'] = 'LLAT'
            output_df.loc[output_df['diag_class'] == -1, 'diag_class'] = 'RLAT'
        if view == 'LA':
            output_df.loc[output_df['diag_class'] == 0, 'diag_class'] = 'NORM'
            output_df.loc[output_df['diag_class'] == 1, 'diag_class'] = 'ANTE'
            output_df.loc[output_df['diag_class'] == -1, 'diag_class'] = 'RETR'

        # plot graph
        plt.figure(figsize=(fig_width,fig_height))
        plt.subplots_adjust(left=0.1, right=0.99, top=0.98, bottom=0.05)
        sns.violinplot(data=output_df, x = 'y', y = 'x', hue = 'diag_class', dodge=False, alpha=0.5)
        plt.xlabel('Normalized Values')
        plt.ylabel('Features')
        FIGURE_NAME = 'Distribution of Feature Values by Spondylolisthesis Classes (' + view + ' View).png'
        FIGURE_PATH = os.path.join(FIGURES_FOLDER, FIGURE_NAME)
        plt.savefig(FIGURE_PATH)

        # save Bhat CSV
        if view == 'AP':
            cols = ['feature', '“NORM” and “ANTE”', '“NORM” and “RETR”', 'Computational time']
        elif view == 'LA':
            cols = ['feature', '“NORM” and “LLAT”', '“NORM” and “RLAT”', 'Computational time']
        CSV_NAME = 'Bhat Coef (' + view + ' View).csv'
        OUTPUT_CSV_PATH = os.path.join(CSV_FOLDER, CSV_NAME)
        bhat_df = pd.DataFrame(bhat_res, columns = cols)
        bhat_df.to_csv(OUTPUT_CSV_PATH)

        print('--- Results of Bhatthacharyya Coefficients ()' + view + ') --- ')
        print('Analyzed!')
        print('figure -> [' + FIGURE_PATH + ']')
        print('csv -> [' + OUTPUT_CSV_PATH + ']')
        print(bhat_df)
