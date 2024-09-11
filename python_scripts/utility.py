import os
from sklearn.metrics import confusion_matrix
from openai import OpenAI
import warnings
from nltk.stem.porter import *
import torch
import re
import logging
import transformers

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)


class Utility:
    """This class evaluates the accuracy of app features extracted from app reviews by matching them with the human labeled app features"""

    def __init__(self):
        pass

    @staticmethod
    def read_prompt(prompt_type):
        instructions_file_path = "../prompt_data/" + prompt_type + "_prompt.txt"
        system_instruction=""
        with open(instructions_file_path, 'r') as file:
            system_instruction = file.read()

        return system_instruction

    @staticmethod
    def calculate_metrics(conf_matrix, sentiment_class):
        sentiment_classes = ["negative", "positive", "neutral"]
        idx = sentiment_classes.index(sentiment_class)
        true_positives = conf_matrix[idx, idx]
        false_positives = sum(conf_matrix[:, idx]) - true_positives
        false_negatives = sum(conf_matrix[idx, :]) - true_positives
        return true_positives, false_positives, false_negatives

    @staticmethod
    def get_classwise_sentiment_performance(true_sentiments,pred_sentiments):
        sentiment_classes = ["negative", "positive", "neutral"]

        conf_matrix = confusion_matrix(true_sentiments, pred_sentiments, labels=sentiment_classes)
        tps_pos_sentiment, fps_pos_sentiment, fns_pos_sentiment = Utility.calculate_metrics(conf_matrix, "positive")
        tps_neutral_sentiment, fps_neutral_sentiment, fns_neutral_sentiment = Utility.calculate_metrics(conf_matrix, "neutral")
        tps_neg_sentiment, fps_neg_sentiment, fns_neg_sentiment = Utility.calculate_metrics(conf_matrix, "negative")

        return {
                "positive":{"tps": tps_pos_sentiment, "fps": fps_pos_sentiment, "fns": fns_pos_sentiment},
                "neutral":{"tps": tps_neutral_sentiment, "fps": fps_neutral_sentiment, "fns": fns_neutral_sentiment},
                "negative":{"tps": tps_neg_sentiment, "fps": fps_neg_sentiment, "fns": fns_neg_sentiment}
                }

    @staticmethod
    def compute_performance_metrics(tps,fps,fns):
        prec, rec, f1 = 0 , 0 , 0

        # calculate precision, recall, and f1-score with exact matching for summary data
        if fps + tps == 0:
            prec = 0
        else:
            prec = tps / (fps + tps)


        if tps + fns == 0:
            rec = 0
        else:
            rec = tps / (fns+ tps)

        if prec!=0 and rec!=0:
            f1 = 2 * (prec * rec)/(prec + rec)
        else:
            f1 = 0

        return prec,rec,f1


    @staticmethod
    def create_folder(path_to_folder):
        folder_path = path_to_folder

        # Check if the folder does not exist, then create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def get_list_of_features_sentiments(feature_col_data,sentiment):
        lst_features_sentiment=[]

        if isinstance(feature_col_data,str):
                #check for multiple features by splitting it with comma

                lst_features = feature_col_data.split(";")

                for feature in lst_features:
                    lst_features_sentiment.append((feature,sentiment))

        return lst_features_sentiment

    @staticmethod
    def get_true_features_sentiments_as_list_of_tuples(df_row):
        lst_features_sentiments_all=[]
        positive_feature = df_row["Feature (Positive)"]
        neutral_feature = df_row["Feature (Neutral)"]
        negative_feature = df_row["Feature (Negative)"]

        positive_features = Utility.get_list_of_features_sentiments(positive_feature,"positive")
        neutral_features = Utility.get_list_of_features_sentiments(neutral_feature,"neutral")
        negative_features = Utility.get_list_of_features_sentiments(negative_feature,"negative")

        if len(positive_features)!=0:
            lst_features_sentiments_all.extend(positive_features)

        if len(neutral_features)!=0:
            lst_features_sentiments_all.extend(neutral_features)

        if len(negative_features)!=0:
            lst_features_sentiments_all.extend(negative_features)


        return lst_features_sentiments_all

    @staticmethod
    def validate_LLM_output(app_features_with_sentiments):
        lst_app_features_with_sentiments = []

        if str(app_features_with_sentiments)== "[None]" or str(app_features_with_sentiments) == "[none]" or str(app_features_with_sentiments)=="[empty]" or str(app_features_with_sentiments) == "[Empty]":
            return lst_app_features_with_sentiments

        feature_list_only_contain_tuples = False
        # if it is a list, make sure that it is a list of tuples
        if len(app_features_with_sentiments)!=0:
            feature_list_only_contain_tuples=all(isinstance(item, tuple) for item in app_features_with_sentiments)

        # ignore tuple if it doesn't contain feature and sentiment
        if feature_list_only_contain_tuples == True:
            lst_app_features_with_sentiments = [feature_sentiment for feature_sentiment in app_features_with_sentiments if len(feature_sentiment)!=0]
            lst_app_features_with_sentiments = [feature_sentiment for feature_sentiment in lst_app_features_with_sentiments if feature_sentiment[1] is not None and feature_sentiment[0] is not None and feature_sentiment[1] in ["positive","negative","neutral"]]
        else:
            lst_app_features_with_sentiments = []


        return lst_app_features_with_sentiments

    @staticmethod
    def stem_feature(app_feature):
        stemmer = PorterStemmer()
        words_extracted_feature = app_feature.strip().split(" ")
        return ' '.join([stemmer.stem(word) for word in words_extracted_feature])

    @staticmethod
    def get_output_from_OS_model_response(model_id,model_output):
        # Regular expression to extract text between [/INST] and </s>
        pattern = re.compile(r'\[\/INST\](.*?)\<\/s\>', re.DOTALL)

        # Find all matches
        matches = pattern.findall(model_output)

        # Print the matched text
        if matches:
            last_match = matches[-1].strip()
            logging.info(f"Match => {last_match}")
            return last_match
        else:
            logging.info(f"Nothing has been found!")
            return ""

    @staticmethod
    def prompt_OS_chat_model(model_id,chat_model,tokenizer,prompt):
        device = torch.device("cuda")
        input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)
        generated_ids = chat_model.generate(input_ids, max_new_tokens=1000, do_sample=True).to(device)
        outputs = tokenizer.batch_decode(generated_ids)
        logging.info(f"Chat model output=>{outputs[0]}")
        return Utility.get_output_from_OS_model_response(model_id,outputs[0])

    @staticmethod
    def prompt_OpenAI_model(chat_model_name,temp,model_secret_key,prompt):
        client =  OpenAI(api_key=model_secret_key)
        completion = client.chat.completions.create(model=chat_model_name,temperature=temp,messages=prompt)
        return completion.choices[0].message.content

