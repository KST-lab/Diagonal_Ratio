import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

target_features = 'diag_num'
for points_from in ['gt', 'dt']:
    # if points_from == 'dt':
    #     continue
    for VIEW in ['AP', 'LA']:
        # load data
        DATASET_PATH = os.path.join('csv', 'spondylolisthesis_features_' + points_from,'spondy_features_' + points_from + '_' + VIEW + '_train.csv')
        df_train = pd.read_csv(DATASET_PATH,index_col=0)

        DATASET_PATH = os.path.join('csv', 'spondylolisthesis_features_' + points_from,'spondy_features_' + points_from + '_' + VIEW + '_test.csv')
        df_test = pd.read_csv(DATASET_PATH,index_col=0)

        DATASET_PATH = os.path.join('csv', 'spondylolisthesis_features_' + points_from,'spondy_features_' + points_from + '_' + VIEW + '_eval.csv')
        df_eval = pd.read_csv(DATASET_PATH,index_col=0)
        
        df = pd.concat([df_train, df_test, df_eval], axis=0, ignore_index=True)

        n_samples_per_class = min(list(df[target_features].value_counts()))
        classifier_array = ['SVC', 'RF', 'GB']

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

        selected_features = ['DH', 'SA', 'AD', 'PSD_L', 'PSD_R', 'SDL', 'SDR', 'ASD', 'DR', 'diag_num']

        cols_coef = []
        cols_importance = []
        res_svc_coef = []
        res_rf_coef = []
        res_gb_coef = []
        for i in range(len(selected_features[0:-1])):
            cols_coef.append('coef_' + selected_features[i])
            cols_importance.append('importance_' + selected_features[i])

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
            # print(input_df)
            # quit()
            X = input_df.drop(target_features, axis = 1)
            y = input_df[target_features].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

            # svc
            svc_classifier = SVC(kernel='linear', random_state=seed)
            svc_classifier.fit(X_train, y_train)
            y_pred = svc_classifier.predict(X_test)
            overall_accuracy['SVC'] = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True,zero_division=True)
            overall_recall['SVC'] = report['macro avg']['recall']
            overall_precision['SVC'] = report['macro avg']['precision']
            overall_f1_score['SVC'] = report['macro avg']['f1-score']
            coefficients = svc_classifier.coef_
            intercept = svc_classifier.intercept_
            Coefficients_df = pd.DataFrame(coefficients)
            mean_abs = Coefficients_df.abs().mean(axis=0)
            row_coef = mean_abs.to_list()
            res_svc_coef.append(row_coef)

            # random forest
            clf = RandomForestClassifier(n_estimators=100, random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            overall_accuracy['RF'] = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True,zero_division=True)
            overall_recall['RF'] = report['macro avg']['recall']
            overall_precision['RF'] = report['macro avg']['precision']
            overall_f1_score['RF'] = report['macro avg']['f1-score']
            importances = clf.feature_importances_
            res_rf_coef.append(importances)


            # gradient boosting
            clf = GradientBoostingClassifier(n_estimators=100, random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            overall_accuracy['GB'] = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True,zero_division=True)
            overall_recall['GB'] = report['macro avg']['recall']
            overall_precision['GB'] = report['macro avg']['precision']
            overall_f1_score['GB'] = report['macro avg']['f1-score']
            importances = clf.feature_importances_
            res_gb_coef.append(importances)    

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
        coef_svc_df = pd.DataFrame(res_svc_coef, columns=cols_coef)
        coef_rf_df = pd.DataFrame(res_rf_coef, columns=cols_importance)
        coef_gb_df = pd.DataFrame(res_gb_coef, columns=cols_importance)

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

        output_df = pd.DataFrame(output_res, columns=cols)

        points_from_map = {
            'gt': 'Ground Truth Points',
            'dt': 'ResNet Points'
        }
        print('--- Results ' + points_from_map[points_from] + ' (' + VIEW + ') --- factor analysis from multiple features classification ---')
        print(' 1. Performance metrics of spondylolisthesis classification:')
        print(output_df)
        print(' 2. Factor Analysis:')
        importance_df = pd.DataFrame([selected_features[0:-1], list(coef_svc_df.mean(axis=0)), coef_rf_df.mean(axis=0), coef_gb_df.mean(axis=0)])
        importance_df = importance_df.transpose()
        importance_df.columns = ['feature', 'SVC Coefficient', 'RF importance', 'GB importance']
        OUTPUT_CSV_FOLDER = os.path.join('csv', 'factor_analysis')
        if not os.path.exists(OUTPUT_CSV_FOLDER):
            os.makedirs(OUTPUT_CSV_FOLDER, exist_ok=True)
        OUTPUT_CSV_FILE_NAME = 'factor_analysis_' + points_from + '_' + VIEW + '.csv'
        OUTPUT_CSV_FILE_PATH = os.path.join(OUTPUT_CSV_FOLDER, OUTPUT_CSV_FILE_NAME)
        importance_df.to_csv(OUTPUT_CSV_FILE_PATH)
        print(importance_df)
        print('output csv path : ' + OUTPUT_CSV_FILE_PATH)
        print('---------------------------------------------------------')



