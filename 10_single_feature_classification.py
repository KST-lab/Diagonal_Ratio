import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB

import numpy as np
def are_values_unique(d):
    seen_values = set()
    for value in d.values():
        if value in seen_values:
            return False  # Found a duplicate value
        seen_values.add(value)
    return True  # All values are unique

target_features = 'diag_num'
for points_from in ['gt', 'dt']:
    for VIEW in ['AP', 'LA']:

        # load data
        DATASET_PATH = os.path.join('csv', 'spondylolisthesis_features_'+points_from,'spondy_features_'+points_from+'_' + VIEW + '_train.csv')
        df_train = pd.read_csv(DATASET_PATH,index_col=0)

        DATASET_PATH = os.path.join('csv', 'spondylolisthesis_features_'+points_from,'spondy_features_'+points_from+'_' + VIEW + '_test.csv')
        df_test = pd.read_csv(DATASET_PATH,index_col=0)

        DATASET_PATH = os.path.join('csv', 'spondylolisthesis_features_'+points_from,'spondy_features_'+points_from+'_' + VIEW + '_eval.csv')
        df_eval = pd.read_csv(DATASET_PATH,index_col=0)
        
        df = pd.concat([df_train, df_test, df_eval], axis=0, ignore_index=True)

        n_samples_per_class = min(list(df[target_features].value_counts()))
        features_array = ['DH', 'SA', 'AD', 'PSD_L', 'PSD_R', 'SDL', 'SDR', 'ASD', 'DR']
        classifier_array = ['NB']

        Features_res = []
        for feature in features_array:
            Features_row = [feature]
            
            accuracy_res = []
            accuracy_cols = ['seed']

            precision_res = []
            precision_cols = ['seed']

            recall_res = []
            recall_cols = ['seed']

            F1_res = []
            F1_cols = ['seed']
            for classifier in classifier_array:
                accuracy_cols = accuracy_cols + [classifier + '_' + '_accuracy']
                precision_cols = precision_cols + [classifier + '_' + '_precision']
                recall_cols = recall_cols + [classifier + '_' + '_recall']
                F1_cols = F1_cols + [classifier + '_' + '_F1']

            selected_features = [feature, 'diag_num']
            cols_coef = []
            res_svm_coef = []
            res_rf_coef = []
            res_gb_coef = []
            for i in range(len(selected_features[0:-1])):
                cols_coef.append('coef_' + selected_features[i])

            for seed in range(100):
                df_class_balanced = df.groupby(target_features).apply(lambda x: x.sample(n=n_samples_per_class, random_state=seed, replace=True)).reset_index(drop=True)

                row = [seed]
                accuracy_row = [seed]
                accuracy_dict = {}
                precision_row = [seed]
                precision_dict = {}
                recall_row = [seed]
                recall_dict = {}
                F1_row = [seed]
                F1_dict = {}
                for classifier in classifier_array:
                    accuracy_dict[classifier] = []
                    precision_dict[classifier] = []
                    recall_dict[classifier] = []
                    F1_dict[classifier] = []

                
                overall_accuracy = {}
                overall_recall = {}
                overall_precision = {}
                overall_f1_score = {}

                input_df = pd.DataFrame()
                input_df = df_class_balanced[selected_features]

                X = input_df.drop(target_features, axis = 1)
                y = input_df[target_features].astype(int)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

                # GaussianNB
                nb_classifier = GaussianNB()
                nb_classifier.fit(X_train, y_train)
                y_pred = nb_classifier.predict(X_test)
                overall_accuracy['NB'] = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True,zero_division=True)
                overall_recall['NB'] = report['macro avg']['recall']
                overall_precision['NB'] = report['macro avg']['precision']
                overall_f1_score['NB'] = report['macro avg']['f1-score']

                for classifier in classifier_array:
                    accuracy_dict[classifier] = accuracy_dict[classifier] + [overall_accuracy[classifier]]
                    precision_dict[classifier] = precision_dict[classifier] + [overall_precision[classifier]]
                    recall_dict[classifier] = recall_dict[classifier] + [overall_recall[classifier]]
                    F1_dict[classifier] = F1_dict[classifier] + [overall_f1_score[classifier]]

                for classifier in classifier_array:
                    accuracy_row = accuracy_row + accuracy_dict[classifier]
                    precision_row = precision_row + precision_dict[classifier]
                    recall_row = recall_row + recall_dict[classifier]
                    F1_row = F1_row + F1_dict[classifier]
                accuracy_res.append(accuracy_row)
                precision_res.append(precision_row)
                recall_res.append(recall_row)
                F1_res.append(F1_row)

            accuracy_res_df = pd.DataFrame(accuracy_res, columns=accuracy_cols)
            precision_res_df = pd.DataFrame(precision_res, columns=precision_cols)
            recall_res_df = pd.DataFrame(recall_res, columns=recall_cols)
            F1_res_df = pd.DataFrame(F1_res, columns=F1_cols)
            
            cols = ['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
            output_res = []
            for classifier in classifier_array:
                accuracy = accuracy_res_df[classifier + '__accuracy'].mean()
                precision = precision_res_df[classifier + '__precision'].mean()
                recall = recall_res_df[classifier + '__recall'].mean()
                F1_score = F1_res_df[classifier + '__F1'].mean()
                output_row = [classifier, accuracy, precision, recall, F1_score]
                output_res.append(output_row)
                Features_row = Features_row + [round(accuracy*100,2), round(precision*100,2)] 
            Features_res.append(Features_row)


            output_df = pd.DataFrame(output_res, columns=cols)

        points_from_map = {
            'gt': 'Ground Truth Points',
            'dt': 'ResNet Points'
        }
        print('--- ' + points_from_map[points_from] + ' (' + VIEW + ') --- confusion metrics from single feature classification ---')
        features_cols = ['classifier', 'accuracy', 'precision', 'recall', 'F1_score']
        features_cols = ['classifier', 'accuracy', 'precision']
        Features_df = pd.DataFrame(Features_res, columns=features_cols)
        OUTPUT_CSV_FOLDER = os.path.join('csv', 'single_feature_classification')
        if not os.path.exists(OUTPUT_CSV_FOLDER):
            os.makedirs(OUTPUT_CSV_FOLDER, exist_ok=True)
        OUTPUT_CSV_FILE_NAME = 'NB_' + points_from + '_' + VIEW + '.csv'
        OUTPUT_CSV_FILE_PATH = os.path.join(OUTPUT_CSV_FOLDER, OUTPUT_CSV_FILE_NAME)
        Features_df.to_csv(OUTPUT_CSV_FILE_PATH)
        print(Features_df)
        print('output csv path : ' + OUTPUT_CSV_FILE_PATH)
