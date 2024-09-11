#!/usr/bin/env python
# coding: utf-8

import copy

class Evaluation:
    """This class evaluates the accuracy of accuracy of features and sentiments identified from app reviews by matching them with the human labeled app features"""
    def __init__(self):
        pass
    @staticmethod
    def feature_sentiment_matching(lst_true_features_sentiments,lst_extracted_features_sentiments,diff_threshold=0):
        sorted_list_true_features,lst_true_features_copy = [],[]
        sorted_list_extracted_features,lst_extracted_features_copy = [], []

        lst_true_features = [feature_sentiment[0] for feature_sentiment in lst_true_features_sentiments]

        dict_true_features_sentiments = {feature_sentiment[0].strip().lower():feature_sentiment[1].strip().lower() for feature_sentiment in lst_true_features_sentiments}

        if len(lst_true_features)!=0:
                sorted_list_true_features = sorted(lst_true_features, key=lambda x: len(x.split()))
                lst_true_features_copy = [feature.strip().lower() for feature in lst_true_features]

        lst_extracted_features = [feature_sentiment[0] for feature_sentiment in lst_extracted_features_sentiments]
        dict_extracted_features_sentiments = {feature_sentiment[0].strip().lower():feature_sentiment[1].strip().lower() for feature_sentiment in lst_extracted_features_sentiments}


        if len(lst_extracted_features)!=0:
            sorted_list_extracted_features = sorted(lst_extracted_features, key=lambda x: len(x.split()))
            lst_extracted_features_copy = [feature.strip().lower() for feature in lst_extracted_features]

        tps_features= 0
        fps_features= 0
        fns_features= 0

        true_sentiments = []
        pred_sentiments = []

        # assumption is that features are unique in both lists (extracted and true)
        for extracted_app_feature in sorted_list_extracted_features:
            for true_app_feature in sorted_list_true_features:
                # determine the number of words in both extracted and true features
                words_extracted_feature = extracted_app_feature.strip().lower().split(" ")
                words_true_feature = true_app_feature.strip().lower().split(" ")
                feature_difference = abs(len(words_extracted_feature) - len(words_true_feature))

                # simple case: both extracted and true features of same size
                if feature_difference==0 and extracted_app_feature.strip().lower() in lst_extracted_features_copy and true_app_feature.strip().lower() in lst_true_features_copy:
                    if extracted_app_feature.strip().lower() == true_app_feature.strip().lower():
                        tps_features = tps_features + 1
                        lst_extracted_features_copy.remove(extracted_app_feature.strip().lower())
                        lst_true_features_copy.remove(true_app_feature.strip().lower())

                        # check for sentiment polarity for the matched feature
                        if dict_true_features_sentiments[true_app_feature.strip().lower()] == dict_extracted_features_sentiments[extracted_app_feature.strip().lower()]:
                            true_sentiments.append(dict_true_features_sentiments[true_app_feature.strip().lower()])
                            pred_sentiments.append(dict_extracted_features_sentiments[extracted_app_feature.strip().lower()])
                        else:
                            true_sentiments.append(dict_true_features_sentiments[true_app_feature.strip().lower()])
                            pred_sentiments.append(dict_extracted_features_sentiments[extracted_app_feature.strip().lower()])

                        #print("TP [","extracted feature=",extracted_app_feature.strip().lower()," true features=",true_app_feature.strip().lower(), "]")
                        break
                elif feature_difference<=diff_threshold and extracted_app_feature.strip().lower() in lst_extracted_features_copy and true_app_feature.strip().lower() in lst_true_features_copy:
                    if len(words_extracted_feature) < len(words_true_feature) and all([feature_word in words_true_feature for feature_word in words_extracted_feature])==True:
                        tps_features = tps_features + 1
                        lst_extracted_features_copy.remove(extracted_app_feature.strip().lower())
                        lst_true_features_copy.remove(true_app_feature.strip().lower())

                        if dict_true_features_sentiments[true_app_feature.strip().lower()] == dict_extracted_features_sentiments[extracted_app_feature.strip().lower()]:
                            true_sentiments.append(dict_true_features_sentiments[true_app_feature.strip().lower()])
                            pred_sentiments.append(dict_extracted_features_sentiments[extracted_app_feature.strip().lower()])
                        else:
                            true_sentiments.append(dict_true_features_sentiments[true_app_feature.strip().lower()])
                            pred_sentiments.append(dict_extracted_features_sentiments[extracted_app_feature.strip().lower()])
                        break
                    elif len(words_extracted_feature) > len(words_true_feature) and all([feature_word in words_extracted_feature for feature_word in words_true_feature])==True:
                        tps_features = tps_features + 1
                        lst_extracted_features_copy.remove(extracted_app_feature.strip().lower())
                        lst_true_features_copy.remove(true_app_feature.strip().lower())

                        if dict_true_features_sentiments[true_app_feature.strip().lower()] == dict_extracted_features_sentiments[extracted_app_feature.strip().lower()]:
                            true_sentiments.append(dict_true_features_sentiments[true_app_feature.strip().lower()])
                            pred_sentiments.append(dict_extracted_features_sentiments[extracted_app_feature.strip().lower()])
                        else:
                            true_sentiments.append(dict_true_features_sentiments[true_app_feature.strip().lower()])
                            pred_sentiments.append(dict_extracted_features_sentiments[extracted_app_feature.strip().lower()])

                        break

        fps_features = len(lst_extracted_features_copy)
        fns_features = len(lst_true_features_copy)

        return {"features":{"tps":tps_features, "fps":fps_features, "fns": fns_features},"sentiments":{"true": true_sentiments, "pred": pred_sentiments}}


def main():
    true_feature_sentiments = [('open','neutral'),('image','neutral')]
    pred_feature_sentiments = [('open', 'negative'), ('upload image', 'neutral'), ('no issues', 'positive'), ('every single day', 'neutral')]

    expected_results={"features":{"tps":2,"fps":2,"fns":0},"sentiments":{"tps":0,"fps":0,"fns":0}}

    eval_results = Evaluation.feature_sentiment_matching(true_feature_sentiments,pred_feature_sentiments,1)
    print(eval_results)
    print(expected_results)

    assert eval_results["features"]["tps"]==expected_results["features"]["tps"]
    assert eval_results["features"]["fps"]==expected_results["features"]["fps"]
    assert eval_results["features"]["fns"]==expected_results["features"]["fns"]

if __name__ == "__main__":
    main()