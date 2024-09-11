from evaluation import Evaluation
import pandas as pd
from transformers import set_seed
from utility import Utility
import numpy as np
import json
import re
import ast
from time import time,sleep
from sklearn.metrics import precision_recall_fscore_support
import timeit
import logging
import os
import argparse
import datetime
import random
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

set_seed(42)

class  Feature_Level_Sentiment_Analysis:

    def __init__(self, model_name,temperature,secret_key,num_of_iterations,ptype,no_of_examples,few_shot_examples_file_path):
        self.model = model_name
        self.temperature = temperature
        self.MODEL_SECRET_KEY = secret_key
        self.ITERATIONS = num_of_iterations
        self.prompt_type = ptype
        self.num_few_shot_examples = no_of_examples
        self.few_shot_examples_filepath= few_shot_examples_file_path

        logging.info(f"The model is set to {self.model}")
        if any([model in self.model.lower() for model in ["mistral","llama","gemma"]]):
            self.OS_chat_model = AutoModelForCausalLM.from_pretrained(self.model,token=self.MODEL_SECRET_KEY,temperature=self.temperature,device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model,token=self.MODEL_SECRET_KEY)

    def set_review_data_with_labels(self,ground_truth_review_data):
        self.labeled_review_data = ground_truth_review_data

    def set_log_data_path(self,log_data_path):
        self.log_data_path= log_data_path

    def set_stemmer_value(self,stemmer_choice):
        self.stemmer_choice = stemmer_choice

    def extract_app_features_with_sentiments(self,system_instruction,user_review):

        review_text = "Review: \"" + user_review + "\""

        prompt = []
        prompt.append({"role":"system","content" : system_instruction})

        #few-shot examples

        if self.num_few_shot_examples>0:
            with open(self.few_shot_examples_filepath, 'r') as file:
                # Load the JSON data
                few_shot_examples = json.load(file)

            if self.num_few_shot_examples <= len(few_shot_examples):
                random_indexes = random.sample(range(len(few_shot_examples)), self.num_few_shot_examples)
                for index in range(len(few_shot_examples)):
                    if index in random_indexes:
                        example = few_shot_examples[index]
                        review_sentence = "Review:\"" + example["Review"] + "\""
                        output = []
                        # Convert each tuple to a formatted string

                        if example["features"]!="none":
                            features = example["features"].split(";")
                            sentiments = example["sentiments"].split(";")

                            for feature, sentiment in zip(features,sentiments):
                                output.append((feature,sentiment))

                        formatted_strings = [f"('{feature}', '{sentiment}')" for feature, sentiment in output]

                        # Join the formatted strings into a single string
                        output_result_string = '[' + ', '.join(formatted_strings) + ']'

                        prompt.append({"role":"user","content": review_sentence})
                        prompt.append({"role":"assistant","content": output_result_string})
            else:
                logging.info("The number of few shot examples asked has exceeded the examples in the json file!")
                sys.exit()

        prompt.append({"role":"user","content" : review_text})

        logging.info(f"Prompt-> {prompt}")

        retry = 0
        max_retry = 5
        while True:
            try:
                if any([model in self.model.lower() for model in ["llama","mistral","gemma"]]):
                    extracted_feature_sentiments = Utility.prompt_OS_chat_model(self.model,self.OS_chat_model,self.tokenizer,prompt)
                else:
                    extracted_feature_sentiments = Utility.prompt_OpenAI_model(self.model,self.temperature,self.MODEL_SECRET_KEY,prompt)

                return extracted_feature_sentiments
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    raise Exception("LLM model error (retries exceed than 5): %s" % oops)
                logging.error("Error communicating with LLM model: ",oops)
                sleep(1)

    def Prompt_LLM_to_Extract_App_Features_with_Sentiments_from_a_Review(self,instruction,review_text):
        output = self.extract_app_features_with_sentiments(instruction,review_text)
        logging.info(f"Review text -> {review_text}")
        logging.info(f"LLM extracted features -> {output}")
        extraction_error = True

        app_features_with_sentiments = []

        # Validate LLM output
        if isinstance(output,str)==True:
            try:
                llm_output =  ast.literal_eval(output)
                app_features_with_sentiments = Utility.validate_LLM_output(llm_output)
                extraction_error = False
            except Exception as ex:
                logging.info(f"Can't find app features and sentiment in the {output}")

        if extraction_error==True and isinstance(output,list) == False:
            pattern = r'\[.*?\]'
            matches = re.findall(pattern, str(output))
            logging.info(f"Regular expression matches -> {matches}")

            try:
                if matches[0].strip() == "[None]" or matches[0].strip() == "[none]" or matches[0].strip()=="[empty]" or matches[0].strip() == "[Empty]":
                    app_features_with_sentiments = []
                    return app_features_with_sentiments
                else:
                    llm_output = eval(matches[0])
                    app_features_with_sentiments = Utility.validate_LLM_output(llm_output)
            except Exception as ex:
                logging.info(f"Can't find app features and sentiment in the {output}")

        return app_features_with_sentiments

    def Eval_LLMs_n_Prompts_on_Review_Data(self):

        dict_true_features_sentiments={}
        # group data by apps
        reviews_group_by_app = self.labeled_review_data.groupby("App id")

        for index, row in self.labeled_review_data.iterrows():
            dict_true_features_sentiments[index] = Utility.get_true_features_sentiments_as_list_of_tuples(row)

        # zero-shot prompt to extract features and sentiments with one prompt
        logging.info(f"Reading prompt type {self.prompt_type}")
        system_instruction = Utility.read_prompt(self.prompt_type)

        # Create an empty DataFrame with columns
        column_names_overall_eval_summary = ["Prec_Exact_Match_Feature","Rec_Exact_Match_Feature","F1_Exact_Match_Feature","Prec_Exact_Match_Sentiment","Rec_Exact_Match_Sentiment","F1_Exact_Match_Sentiment","Prec_Partial_Match_One_Feature","Rec_Partial_Match_One_Feature","F1_Partial_Match_One_Feature","Prec_Partial_Match_One_Sentiment","Rec_Partial_Match_One_Sentiment","F1_Partial_Match_One_Sentiment","Prec_Partial_Match_Two_Feature","Rec_Partial_Match_Two_Feature","F1_Partial_Match_Two_Feature","Prec_Partial_Match_Two_Sentiment","Rec_Partial_Match_Two_Sentiment","F1_Partial_Match_Two_Sentiment","Execution time (in seconds)"]
        column_names_sentiment_performance= ["Prec_Positive","Rec_Positive","F1_Positive","Prec_Neutral","Rec_Neutral","F1_Neutral","Prec_Negative","Rec_Negative","F1_Negative"]

        df_prompt_eval_summary_log_data_app_wise = pd.DataFrame(columns=["App_ID"] + column_names_overall_eval_summary)
        df_prompt_eval_sentiment_summary_app_wise_exact_match = pd.DataFrame(columns=["App_ID"] + column_names_sentiment_performance)
        df_prompt_eval_sentiment_summary_app_wise_partial_match_one = pd.DataFrame(columns= ["App_ID"] + column_names_sentiment_performance)
        df_prompt_eval_sentiment_summary_app_wise_partial_match_two = pd.DataFrame(columns= ["App_ID"] + column_names_sentiment_performance)

        df_prompt_eval_overall_summary_log_data = pd.DataFrame(columns=column_names_overall_eval_summary)

        df_prompt_eval_overall_senti_eval_summary_exact_match = pd.DataFrame(columns=column_names_sentiment_performance)
        df_prompt_eval_overall_senti_eval_summary_partial_match_one = pd.DataFrame(columns=column_names_sentiment_performance)
        df_prompt_eval_overall_senti_eval_summary_partial_match_two = pd.DataFrame(columns=column_names_sentiment_performance)

        Utility.create_folder(self.log_data_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        learning_type = "few_shot_" + str(self.num_few_shot_examples) if self.num_few_shot_examples>0 else "zero_shot"
        output_folder_name = f"{self.model}_{self.temperature}_{self.ITERATIONS}_{self.prompt_type}_{learning_type}_{timestamp}"
        model_path_folder =  self.log_data_path + output_folder_name

        Utility.create_folder(model_path_folder)

        lst_prec_exact_match_overall,lst_rec_exact_match_overall,lst_f1_exact_overall = [],[],[]
        lst_prec_exact_match_overall_sentiment,lst_rec_exact_match_overall_sentiment,lst_f1_exact_overall_sentiment = [],[],[]
        lst_prec_exact_match_overall_pos_sentiment,lst_rec_exact_match_overall_pos_sentiment,lst_f1_exact_overall_pos_sentiment = [],[],[]
        lst_prec_exact_match_overall_neutral_sentiment,lst_rec_exact_match_overall_neutral_sentiment,lst_f1_exact_overall_neutral_sentiment = [],[],[]
        lst_prec_exact_match_overall_neg_sentiment,lst_rec_exact_match_overall_neg_sentiment,lst_f1_exact_overall_neg_sentiment = [],[],[]

        lst_prec_partial_one_word_match_overall,lst_rec_partial_one_word_match_overall,lst_f1_partial_one_word_match_overall = [],[],[]
        lst_prec_partial_one_word_match_overall_sentiment,lst_rec_partial_one_word_match_overall_sentiment,lst_f1_partial_one_word_match_overall_sentiment = [],[],[]
        lst_prec_partial_one_word_match_overall_pos_sentiment,lst_rec_partial_one_word_match_overall_pos_sentiment,lst_f1_partial_one_word_match_overall_pos_sentiment = [],[],[]
        lst_prec_partial_one_word_match_overall_neutral_sentiment,lst_rec_partial_one_word_match_overall_neutral_sentiment,lst_f1_partial_one_word_match_overall_neutral_sentiment = [],[],[]
        lst_prec_partial_one_word_match_overall_neg_sentiment,lst_rec_partial_one_word_match_overall_neg_sentiment,lst_f1_partial_one_word_match_overall_neg_sentiment = [],[],[]


        lst_prec_partial_two_word_match_overall,lst_rec_partial_two_word_match_overall,lst_f1_partial_two_word_match_overall = [],[],[]
        lst_prec_partial_two_word_match_overall_sentiment,lst_rec_partial_two_word_match_overall_sentiment,lst_f1_partial_two_word_match_overall_sentiment = [],[],[]
        lst_prec_partial_two_word_match_overall_pos_sentiment,lst_rec_partial_two_word_match_overall_pos_sentiment,lst_f1_partial_two_word_match_overall_pos_sentiment = [],[],[]
        lst_prec_partial_two_word_match_overall_neutral_sentiment,lst_rec_partial_two_word_match_overall_neutral_sentiment,lst_f1_partial_two_word_match_overall_neutral_sentiment = [],[],[]
        lst_prec_partial_two_word_match_overall_neg_sentiment,lst_rec_partial_two_word_match_overall_neg_sentiment,lst_f1_partial_two_word_match_overall_neg_sentiment = [],[],[]
        lst_execution_time_overall = []

        logging.info(f"System instruction-> {system_instruction}")

        for app_id, app_reviews in reviews_group_by_app:

            lst_prec_exact_match_app,lst_rec_exact_match_app,lst_f1_exact_match_app = [],[],[]
            lst_prec_exact_match_app_sentiment,lst_rec_exact_match_app_sentiment,lst_f1_exact_match_app_sentiment = [],[],[]
            lst_prec_exact_match_app_pos_sentiment,lst_rec_exact_match_app_pos_sentiment,lst_f1_exact_match_app_pos_sentiment = [],[],[]
            lst_prec_exact_match_app_neutral_sentiment,lst_rec_exact_match_app_neutral_sentiment,lst_f1_exact_match_app_neutral_sentiment = [],[],[]
            lst_prec_exact_match_app_neg_sentiment,lst_rec_exact_match_app_neg_sentiment,lst_f1_exact_match_app_neg_sentiment = [],[],[]

            lst_prec_partial_one_word_match_app,lst_rec_partial_one_word_match_app,lst_f1_partial_one_word_match_app = [], [], []
            lst_prec_partial_one_word_match_app_sentiment,lst_rec_partial_one_word_match_app_sentiment,lst_f1_partial_one_word_match_app_sentiment = [], [], []
            lst_prec_partial_one_word_match_app_pos_sentiment,lst_rec_partial_one_word_match_app_pos_sentiment,lst_f1_partial_one_word_match_app_pos_sentiment = [], [], []
            lst_prec_partial_one_word_match_app_neutral_sentiment,lst_rec_partial_one_word_match_app_neutral_sentiment,lst_f1_partial_one_word_match_app_neutral_sentiment = [], [], []
            lst_prec_partial_one_word_match_app_neg_sentiment,lst_rec_partial_one_word_match_app_neg_sentiment,lst_f1_partial_one_word_match_app_neg_sentiment = [], [], []

            lst_prec_partial_two_word_match_app,lst_rec_partial_two_word_match_app,lst_f1_partial_two_word_match_app = [], [], []
            lst_prec_partial_two_word_match_app_sentiment,lst_rec_partial_two_word_match_app_sentiment,lst_f1_partial_two_word_match_app_sentiment = [], [], []
            lst_prec_partial_two_word_match_app_pos_sentiment,lst_rec_partial_two_word_match_app_pos_sentiment,lst_f1_partial_two_word_match_app_pos_sentiment = [], [], []
            lst_prec_partial_two_word_match_app_neutral_sentiment,lst_rec_partial_two_word_match_app_neutral_sentiment,lst_f1_partial_two_word_match_app_neutral_sentiment = [], [], []
            lst_prec_partial_two_word_match_app_neg_sentiment,lst_rec_partial_two_word_match_app_neg_sentiment,lst_f1_partial_two_word_match_app_neg_sentiment = [], [], []
            lst_execution_time_app = []

            columns = ['Review_Sentence', 'True_Features','Extracted_Features', 'True_Sentiments',"Pred_Sentiments","TPs_Exact_Match","FPs_Exact_Match","FNs_Exact_Match","TPs_Partial_Match_One","FPs_Partial_Match_One","FNs_Partial_Match_One","TPs_Partial_Match_Two","FPs_Partial_Match_Two","FNs_Partial_Match_Two"]
            df_feature_extraction_n_eval_detailed_log_data_app = pd.DataFrame(columns=columns)

            logging.info(f"Start processing reviews of {app_id}")

            for iteration in range(1,self.ITERATIONS+1):

                iteration_folder_path = model_path_folder + "/" + str(iteration)
                Utility.create_folder(iteration_folder_path)

                total_tps_exact_app, total_fps_exact_app, total_fns_exact_app = 0, 0, 0
                lst_true_exact_app_sentiment, lst_pred_exact_app_sentiment = [],[]

                total_tps_partial_one_word_app, total_fps_partial_one_word_app, total_fns_partial_one_word_app = 0, 0 ,0
                lst_true_partial_one_word_app_sentiment, lst_pred_partial_one_word_app_sentiment = [], []

                total_tps_partial_two_word_app, total_fps_partial_two_word_app, total_fns_partial_two_word_app = 0, 0 ,0
                lst_true_partial_two_word_app_sentiment, lst_pred_partial_two_word_app_sentiment =  [],[]

                start_time_app = timeit.default_timer()

                logging.info(f"ITERATION#{iteration}")

                for index, row in app_reviews.iterrows():

                    review_sentence = row['Sentence content'].strip()
                    true_features_sentiments = dict_true_features_sentiments[index]
                    true_features = [feature_sentiment[0] for feature_sentiment in true_features_sentiments]

                    if self.stemmer_choice==1 and len(true_features_sentiments)>0:
                        true_features = [Utility.stem_feature(feature_sentiment[0]) for feature_sentiment in true_features_sentiments]

                    true_sentiments = [feature_sentiment[1] for feature_sentiment in true_features_sentiments]

                    logging.info(f"True features=> {true_features}")

                    # prompt LLM to extract features, pass prompt (instruction and query) with input review sentence
                    extracted_features_sentiments= self.Prompt_LLM_to_Extract_App_Features_with_Sentiments_from_a_Review(system_instruction,review_sentence)
                    extracted_features = [feature_sentiment[0] for feature_sentiment in extracted_features_sentiments]
                    extracted_sentiments = [feature_sentiment[1] for feature_sentiment in extracted_features_sentiments]

                    if self.stemmer_choice==1 and len(extracted_features_sentiments)>0:
                        extracted_features = [Utility.stem_feature(feature_sentiment[0]) for feature_sentiment in extracted_features_sentiments]

                    eval_results_exact_match= Evaluation.feature_sentiment_matching(true_features_sentiments,extracted_features_sentiments,0)
                    tps_exact,fps_exact,fns_exact = eval_results_exact_match["features"]["tps"],eval_results_exact_match["features"]["fps"],eval_results_exact_match["features"]["fns"]
                    exact_sentiment_true,exact_sentiment_pred = eval_results_exact_match["sentiments"]["true"],eval_results_exact_match["sentiments"]["pred"]

                    eval_results_exact_partial_one_word = Evaluation.feature_sentiment_matching(true_features_sentiments,extracted_features_sentiments,1)
                    tps_partial_one_word,fps_partial_one_word,fns_partial_one_word = eval_results_exact_partial_one_word["features"]["tps"],eval_results_exact_partial_one_word["features"]["fps"],eval_results_exact_partial_one_word["features"]["fns"]
                    partial_one_word_sentiment_true,partial_one_word_sentiment_pred = eval_results_exact_partial_one_word["sentiments"]["true"],eval_results_exact_partial_one_word["sentiments"]["pred"]

                    eval_results_exact_partial_two_word = Evaluation.feature_sentiment_matching(true_features_sentiments,extracted_features_sentiments,2)
                    tps_partial_two_word,fps_partial_two_word,fns_partial_two_word = eval_results_exact_partial_two_word["features"]["tps"],eval_results_exact_partial_two_word["features"]["fps"],eval_results_exact_partial_two_word["features"]["fns"]
                    partial_two_word_sentiment_true,partial_two_word_sentiment_pred = eval_results_exact_partial_two_word["sentiments"]["true"],eval_results_exact_partial_two_word["sentiments"]["pred"]

                    # micro sum for exact feature matching
                    total_tps_exact_app = total_tps_exact_app + tps_exact
                    total_fps_exact_app = total_fps_exact_app + fps_exact
                    total_fns_exact_app = total_fns_exact_app + fns_exact

                    # micro sum for partial matching (by difference of one word)
                    total_tps_partial_one_word_app = total_tps_partial_one_word_app + tps_partial_one_word
                    total_fps_partial_one_word_app = total_fps_partial_one_word_app + fps_partial_one_word
                    total_fns_partial_one_word_app = total_fns_partial_one_word_app + fns_partial_one_word

                    # micro sum for partial matching (by difference of two word)
                    total_tps_partial_two_word_app = total_tps_partial_two_word_app + tps_partial_two_word
                    total_fps_partial_two_word_app = total_fps_partial_two_word_app + fps_partial_two_word
                    total_fns_partial_two_word_app = total_fns_partial_two_word_app + fns_partial_two_word

                    lst_true_exact_app_sentiment.extend(exact_sentiment_true); lst_pred_exact_app_sentiment.extend(exact_sentiment_pred)
                    lst_true_partial_one_word_app_sentiment.extend(partial_one_word_sentiment_true); lst_pred_partial_one_word_app_sentiment.extend(partial_one_word_sentiment_pred)
                    lst_true_partial_two_word_app_sentiment.extend(partial_two_word_sentiment_true); lst_pred_partial_two_word_app_sentiment.extend(partial_two_word_sentiment_pred)

                    df_feature_extraction_n_eval_detailed_log_data_app = pd.concat([df_feature_extraction_n_eval_detailed_log_data_app,pd.DataFrame([{'Review_Sentence': review_sentence, 'True_Features': true_features, 'Extracted_Features': extracted_features,'True_Sentiments': true_sentiments, 'Pred_Sentiments': extracted_sentiments,'TPs_Exact_Match': tps_exact,'FPs_Exact_Match':fps_exact,'FNs_Exact_Match':fns_exact,"TPs_Partial_Match_One":tps_partial_one_word,"FPs_Partial_Match_One":fps_partial_one_word,"FNs_Partial_Match_One":fns_partial_one_word,"TPs_Partial_Match_Two":tps_partial_two_word,"FPs_Partial_Match_Two":fps_partial_two_word,"FNs_Partial_Match_Two":fns_partial_two_word}])], ignore_index=True)


                end_time_app = timeit.default_timer()

                execution_time_app =  end_time_app - start_time_app

                lst_execution_time_app.append(execution_time_app)

                # save extraction and evaluation details in a csv file
                folder_path_individual_app_feature_sentiment_preds_log = iteration_folder_path + "/" + "individual_app_feature_sentiment_preds_log"
                Utility.create_folder(iteration_folder_path + "/" + "individual_app_feature_sentiment_preds_log")
                detailed_log_eval_file_name_app = folder_path_individual_app_feature_sentiment_preds_log + "/" + "feature_extraction_eval_detailed_log_" + str(app_id) + ".csv"
                df_feature_extraction_n_eval_detailed_log_data_app.to_csv(detailed_log_eval_file_name_app, index=False)

                logging.info(f"Detailed evaluation Log of app {app_id} extracted features against the prompt has been saved!!")

                # # calculate precision, recall, and f1-score with differnt matching strategies for summary data (feature and sentiment)

                eval_exact_match_sentiment = Utility.get_classwise_sentiment_performance(lst_true_exact_app_sentiment,lst_pred_exact_app_sentiment)
                eval_partial_match_one_word_sentiment = Utility.get_classwise_sentiment_performance(lst_true_partial_one_word_app_sentiment,lst_pred_partial_one_word_app_sentiment)
                eval_partial_match_two_word_sentiment = Utility.get_classwise_sentiment_performance(lst_true_partial_two_word_app_sentiment,lst_pred_partial_two_word_app_sentiment)

                prec_exact_match_app,rec_exact_match_app,f1_exact_match_app = Utility.compute_performance_metrics(total_tps_exact_app,total_fps_exact_app,total_fns_exact_app)
                prec_exact_match_app_sentiment, rec_exact_match_app_sentiment, f1_exact_match_app_sentiment,_ = precision_recall_fscore_support(lst_true_exact_app_sentiment,lst_pred_exact_app_sentiment,zero_division=0,average="micro")
                prec_exact_match_app_pos_sentiment,rec_exact_match_app_pos_sentiment,f1_exact_match_app_pos_sentiment = Utility.compute_performance_metrics(eval_exact_match_sentiment["positive"]["tps"],eval_exact_match_sentiment["positive"]["fps"],eval_exact_match_sentiment["positive"]["fns"])
                prec_exact_match_app_neutral_sentiment,rec_exact_match_app_neutral_sentiment,f1_exact_match_app_neutral_sentiment = Utility.compute_performance_metrics(eval_exact_match_sentiment["neutral"]["tps"],eval_exact_match_sentiment["neutral"]["fps"],eval_exact_match_sentiment["neutral"]["fns"])
                prec_exact_match_app_neg_sentiment,rec_exact_match_app_neg_sentiment,f1_exact_match_app_neg_sentiment = Utility.compute_performance_metrics(eval_exact_match_sentiment["negative"]["tps"],eval_exact_match_sentiment["negative"]["fps"],eval_exact_match_sentiment["negative"]["fns"])

                lst_prec_exact_match_app_pos_sentiment.append(prec_exact_match_app_pos_sentiment); lst_rec_exact_match_app_pos_sentiment.append(rec_exact_match_app_pos_sentiment); lst_f1_exact_match_app_pos_sentiment.append(f1_exact_match_app_pos_sentiment)
                lst_prec_exact_match_app_neutral_sentiment.append(prec_exact_match_app_neutral_sentiment); lst_rec_exact_match_app_neutral_sentiment.append(rec_exact_match_app_neutral_sentiment); lst_f1_exact_match_app_neutral_sentiment.append(f1_exact_match_app_neutral_sentiment)
                lst_prec_exact_match_app_neg_sentiment.append(prec_exact_match_app_neg_sentiment); lst_rec_exact_match_app_neg_sentiment.append(rec_exact_match_app_neg_sentiment); lst_f1_exact_match_app_neg_sentiment.append(f1_exact_match_app_neg_sentiment)

                prec_partial_one_word_match_app,rec_partial_one_word_match_app,f1_partial_one_word_match_app = Utility.compute_performance_metrics(total_tps_partial_one_word_app,total_fps_partial_one_word_app,total_fns_partial_one_word_app)
                prec_partial_one_word_match_app_sentiment,rec_partial_one_word_match_app_sentiment,f1_partial_one_word_match_app_sentiment,_=  precision_recall_fscore_support(lst_true_partial_one_word_app_sentiment,lst_pred_partial_one_word_app_sentiment,zero_division=0, average='micro')
                prec_partial_one_word_match_app_pos_sentiment,rec_partial_one_word_match_app_pos_sentiment,f1_partial_one_word_match_app_pos_sentiment = Utility.compute_performance_metrics(eval_partial_match_one_word_sentiment["positive"]["tps"],eval_partial_match_one_word_sentiment["positive"]["fps"],eval_partial_match_one_word_sentiment["positive"]["fns"])
                prec_partial_one_word_match_app_neutral_sentiment,rec_partial_one_word_match_app_neutral_sentiment,f1_partial_one_word_match_app_neutral_sentiment = Utility.compute_performance_metrics(eval_partial_match_one_word_sentiment["neutral"]["tps"],eval_partial_match_one_word_sentiment["neutral"]["fps"],eval_partial_match_one_word_sentiment["neutral"]["fns"])
                prec_partial_one_word_match_app_neg_sentiment,rec_partial_one_word_match_app_neg_sentiment,f1_partial_one_word_match_app_neg_sentiment = Utility.compute_performance_metrics(eval_partial_match_one_word_sentiment["negative"]["tps"],eval_partial_match_one_word_sentiment["negative"]["fps"],eval_partial_match_one_word_sentiment["negative"]["fns"])

                lst_prec_partial_one_word_match_app_pos_sentiment.append(prec_partial_one_word_match_app_pos_sentiment); lst_rec_partial_one_word_match_app_pos_sentiment.append(rec_partial_one_word_match_app_pos_sentiment); lst_f1_partial_one_word_match_app_pos_sentiment.append(f1_partial_one_word_match_app_pos_sentiment)
                lst_prec_partial_one_word_match_app_neutral_sentiment.append(prec_partial_one_word_match_app_neutral_sentiment); lst_rec_partial_one_word_match_app_neutral_sentiment.append(rec_partial_one_word_match_app_neutral_sentiment); lst_f1_partial_one_word_match_app_neutral_sentiment.append(f1_partial_one_word_match_app_neutral_sentiment)
                lst_prec_partial_one_word_match_app_neg_sentiment.append(prec_partial_one_word_match_app_neg_sentiment); lst_rec_partial_one_word_match_app_neg_sentiment.append(rec_partial_one_word_match_app_neg_sentiment); lst_f1_partial_one_word_match_app_neg_sentiment.append(f1_partial_one_word_match_app_neg_sentiment)

                prec_partial_two_word_match_app,rec_partial_two_word_match_app,f1_partial_two_word_match_app = Utility.compute_performance_metrics(total_tps_partial_two_word_app,total_fps_partial_two_word_app,total_fns_partial_two_word_app)
                prec_partial_two_word_match_app_sentiment,rec_partial_two_word_match_app_sentiment,f1_partial_two_word_match_app_sentiment,_ = precision_recall_fscore_support(lst_true_partial_two_word_app_sentiment,lst_pred_partial_two_word_app_sentiment,zero_division=0, average='micro')
                prec_partial_two_word_match_app_pos_sentiment,rec_partial_two_word_match_app_pos_sentiment,f1_partial_two_word_match_app_pos_sentiment = Utility.compute_performance_metrics(eval_partial_match_two_word_sentiment["positive"]["tps"],eval_partial_match_two_word_sentiment["positive"]["fps"],eval_partial_match_two_word_sentiment["positive"]["fns"])
                prec_partial_two_word_match_app_neutral_sentiment,rec_partial_two_word_match_app_neutral_sentiment,f1_partial_two_word_match_app_neutral_sentiment = Utility.compute_performance_metrics(eval_partial_match_two_word_sentiment["neutral"]["tps"],eval_partial_match_two_word_sentiment["neutral"]["fps"],eval_partial_match_two_word_sentiment["neutral"]["fns"])
                prec_partial_two_word_match_app_neg_sentiment,rec_partial_two_word_match_app_neg_sentiment,f1_partial_two_word_match_app_neg_sentiment = Utility.compute_performance_metrics(eval_partial_match_two_word_sentiment["negative"]["tps"],eval_partial_match_two_word_sentiment["negative"]["fps"],eval_partial_match_two_word_sentiment["negative"]["fns"])

                lst_prec_partial_two_word_match_app_pos_sentiment.append(prec_partial_two_word_match_app_pos_sentiment); lst_rec_partial_two_word_match_app_pos_sentiment.append(rec_partial_two_word_match_app_pos_sentiment); lst_f1_partial_two_word_match_app_pos_sentiment.append(f1_partial_two_word_match_app_pos_sentiment)
                lst_prec_partial_two_word_match_app_neutral_sentiment.append(prec_partial_two_word_match_app_neutral_sentiment); lst_rec_partial_two_word_match_app_neutral_sentiment.append(rec_partial_two_word_match_app_neutral_sentiment); lst_f1_partial_two_word_match_app_neutral_sentiment.append(f1_partial_two_word_match_app_neutral_sentiment)
                lst_prec_partial_two_word_match_app_neg_sentiment.append(prec_partial_two_word_match_app_neg_sentiment); lst_rec_partial_two_word_match_app_neg_sentiment.append(rec_partial_two_word_match_app_neg_sentiment); lst_f1_partial_two_word_match_app_neg_sentiment.append(f1_partial_two_word_match_app_neg_sentiment)

                df_feature_performance = pd.DataFrame(
                    {
                        'Prec_exact_match': ["{:.1f}".format(np.mean(prec_exact_match_app)*100)], 'Rec_exact_match':["{:.1f}".format(np.mean(rec_exact_match_app)*100)],'F1_exact_match':[ "{:.1f}".format(np.mean(f1_exact_match_app)*100)],
                        'Prec_partial_match_one_word': ["{:.1f}".format(np.mean(prec_partial_one_word_match_app)*100)],'Rec_partial_match_one_word':["{:.1f}".format(np.mean(rec_partial_one_word_match_app)*100)],'F1_partial_match_one_word':["{:.1f}".format(np.mean(f1_partial_one_word_match_app)*100)],
                        'Prec_partial_match_two_word': [ "{:.1f}".format(np.mean(prec_partial_two_word_match_app)*100)], 'Rec_partial_match_two_word':["{:.1f}".format(np.mean(rec_partial_two_word_match_app)*100)],'F1_partial_match_two_word':["{:.1f}".format(np.mean(f1_partial_two_word_match_app)*100)]
                    }
                )

                df_feature_extraction_confusion_matrix = pd.DataFrame(
                    {
                        'TPs_exact_match': [total_tps_exact_app], 'FPs_exact_match':[total_fps_exact_app],'FNs_exact_match':[total_fns_exact_app],
                        'TPs_partial_match_one_word': [total_tps_partial_one_word_app], 'FPs_partial_match_one_word':[total_fps_partial_one_word_app],'FNs_partial_match_one_word':[total_fns_partial_one_word_app],
                        'TPs_partial_match_two_word': [total_tps_partial_two_word_app], 'FPs_partial_match_two_word':[total_fps_partial_two_word_app],'FNs_partial_match_two_word':[total_fns_partial_two_word_app]
                    }
                )

                df_sentiment_performance_classwise = pd.DataFrame(
                    {
                        'Prec_sentiment_positive_exact': ["{:.1f}".format(np.mean(prec_exact_match_app_pos_sentiment)*100)], 'Rec_sentiment_positive_exact':["{:.1f}".format(np.mean(rec_exact_match_app_pos_sentiment)*100)],'F1_sentiment_positive_exact':["{:.1f}".format(np.mean(f1_exact_match_app_pos_sentiment)*100)],
                        'Prec_sentiment_neutral_exact': [ "{:.1f}".format(np.mean(prec_exact_match_app_neutral_sentiment)*100)], 'Rec_sentiment_neutral_exact':["{:.1f}".format(np.mean(rec_exact_match_app_neutral_sentiment)*100)],'F1_sentiment_neutral_exact':["{:.1f}".format(np.mean(f1_exact_match_app_neutral_sentiment)*100)],
                        'Prec_sentiment_negative_exact': [ "{:.1f}".format(np.mean(prec_exact_match_app_neg_sentiment)*100)], 'Rec_sentiment_negative_exact':["{:.1f}".format(np.mean(rec_exact_match_app_neg_sentiment)*100)],'F1_sentiment_negative_exact':["{:.1f}".format(np.mean(f1_exact_match_app_neg_sentiment)*100)],
                        'Prec_sentiment_positive_partial_one': [ "{:.1f}".format(np.mean(prec_partial_one_word_match_app_pos_sentiment)*100)], 'Rec_sentiment_positive_partial_one':["{:.1f}".format(np.mean(rec_partial_one_word_match_app_pos_sentiment)*100)],'F1_sentiment_positive_partial_one':[ "{:.1f}".format(np.mean(f1_partial_one_word_match_app_pos_sentiment)*100)],
                        'Prec_sentiment_neutral_partial_one': ["{:.1f}".format(np.mean(prec_partial_one_word_match_app_neutral_sentiment)*100)], 'Rec_sentiment_neutral_partial_one':["{:.1f}".format(np.mean(rec_partial_one_word_match_app_neutral_sentiment)*100)],'F1_sentiment_neutral_partial_one':["{:.1f}".format(np.mean(f1_partial_one_word_match_app_neutral_sentiment)*100)],
                        'Prec_sentiment_negative_partial_one': ["{:.1f}".format(np.mean(prec_partial_one_word_match_app_neg_sentiment)*100)], 'Rec_sentiment_negative_partial_one':["{:.1f}".format(np.mean(rec_partial_one_word_match_app_neg_sentiment)*100)],'F1_sentiment_negative_partial_one':["{:.1f}".format(np.mean(f1_partial_one_word_match_app_neg_sentiment)*100)],
                        'Prec_sentiment_positive_partial_two': ["{:.1f}".format(np.mean(prec_partial_two_word_match_app_pos_sentiment)*100)], 'Rec_sentiment_positive_partial_two':["{:.1f}".format(np.mean(rec_partial_two_word_match_app_pos_sentiment)*100)],'F1_sentiment_positive_partial_two':[ "{:.1f}".format(np.mean(f1_partial_two_word_match_app_pos_sentiment)*100)],
                        'Prec_sentiment_neutral_partial_two': [ "{:.1f}".format(np.mean(prec_partial_two_word_match_app_neutral_sentiment)*100)], 'Rec_sentiment_neutral_partial_two':[ "{:.1f}".format(np.mean(rec_partial_two_word_match_app_neutral_sentiment)*100)],'F1_sentiment_neutral_partial_two':["{:.1f}".format(np.mean(f1_partial_two_word_match_app_neutral_sentiment)*100)],
                        'Prec_sentiment_negative_partial_two': ["{:.1f}".format(np.mean(prec_partial_two_word_match_app_neg_sentiment)*100)], 'Rec_sentiment_negative_partial_two':[ "{:.1f}".format(np.mean(rec_partial_two_word_match_app_neg_sentiment)*100)],'F1_sentiment_negative_partial_two':[ "{:.1f}".format(np.mean(f1_partial_two_word_match_app_neg_sentiment)*100)]
                    }
                )

                df_sentiment_performance_metrics_classwise = pd.DataFrame(
                    {
                        'sentiment_true_exact': [lst_true_exact_app_sentiment], 'sentiment_pred_exact':[lst_pred_exact_app_sentiment],
                        'TPs_sentiment_positive_exact': [eval_exact_match_sentiment["positive"]["tps"]], 'FPs_sentiment_positive_exact':[eval_exact_match_sentiment["positive"]["fps"]],'FNs_sentiment_positive_exact':[eval_exact_match_sentiment["positive"]["fns"]],
                        'TPs_sentiment_neutral_exact': [eval_exact_match_sentiment["neutral"]["tps"]], 'FPs_sentiment_neutral_exact':[eval_exact_match_sentiment["neutral"]["fps"]],'FNs_sentiment_neutral_exact':[eval_exact_match_sentiment["neutral"]["fns"]],
                        'TPs_sentiment_negative_exact': [eval_exact_match_sentiment["negative"]["tps"]], 'FPs_sentiment_negative_exact':[eval_exact_match_sentiment["negative"]["fps"]],'FNs_sentiment_negative_exact':[eval_exact_match_sentiment["negative"]["fns"]],
                        'sentiment_true_partial_one': [lst_true_partial_one_word_app_sentiment], 'sentiment_pred_exact_partial_one':[lst_pred_partial_one_word_app_sentiment],
                        'TPs_sentiment_positive_partial_one': [eval_partial_match_one_word_sentiment["positive"]["tps"]], 'FPs_sentiment_positive_partial_one':[eval_partial_match_one_word_sentiment["positive"]["fps"]],'FNs_sentiment_positive_partial_one':[eval_partial_match_one_word_sentiment["positive"]["fns"]],
                        'TPs_sentiment_neutral_partial_one': [eval_partial_match_one_word_sentiment["neutral"]["tps"]], 'FPs_sentiment_neutral_partial_one':[eval_partial_match_one_word_sentiment["neutral"]["fps"]],'FNs_sentiment_neutral_partial_one':[eval_partial_match_one_word_sentiment["neutral"]["fns"]],
                        'TPs_sentiment_negative_partial_one': [eval_partial_match_one_word_sentiment["negative"]["tps"]], 'FPs_sentiment_negative_partial_one':[eval_partial_match_one_word_sentiment["negative"]["fps"]],'FNs_sentiment_negative_partial_one':[eval_partial_match_one_word_sentiment["negative"]["fns"]],
                        'sentiment_true_partial_two': [lst_true_partial_two_word_app_sentiment], 'sentiment_pred_exact_partial_two':[lst_pred_partial_two_word_app_sentiment],
                        'TPs_sentiment_positive_partial_two': [eval_partial_match_two_word_sentiment["positive"]["tps"]], 'FPs_sentiment_positive_partial_two':[eval_partial_match_two_word_sentiment["positive"]["fps"]],'FNs_sentiment_positive_partial_two':[eval_partial_match_two_word_sentiment["positive"]["fns"]],
                        'TPs_sentiment_neutral_partial_two': [eval_partial_match_two_word_sentiment["neutral"]["tps"]], 'FPs_sentiment_neutral_partial_two':[eval_partial_match_two_word_sentiment["neutral"]["fps"]],'FNs_sentiment_neutral_partial_two':[eval_partial_match_two_word_sentiment["neutral"]["fns"]],
                        'TPs_sentiment_negative_partial_two': [eval_partial_match_two_word_sentiment["negative"]["tps"]], 'FPs_sentiment_negative_partial_two':[eval_partial_match_two_word_sentiment["negative"]["fps"]],'FNs_sentiment_negative_partial_two':[eval_partial_match_two_word_sentiment["negative"]["fns"]]
                    }
                )

                confusion_matrx_folder_path = iteration_folder_path + "/" + "individual_app_confusion_matrix_feature_sentiment_preds" + "/"
                Utility.create_folder(confusion_matrx_folder_path)

                performance_eval_folder_path = iteration_folder_path + "/" + "individual_app_performance_metrics_feature_sentiment_preds" + "/"
                Utility.create_folder(performance_eval_folder_path)

                df_feature_extraction_confusion_matrix.to_csv(confusion_matrx_folder_path + "/" + "feature_extraction_cm_" + str(app_id) + ".csv", index=False)
                df_feature_performance.to_csv(performance_eval_folder_path + "/" + "feature_extraction_performance_" + str(app_id) + ".csv", index=False)
                df_sentiment_performance_classwise.to_csv(performance_eval_folder_path + "/" + "classwise_sentiment_performance_" + str(app_id) + ".csv", index=False)
                df_sentiment_performance_metrics_classwise.to_csv(confusion_matrx_folder_path + "/" + "classwise_sentiment_cf_" + str(app_id) + ".csv", index=False)

                # for each app, add precision, recall , f1-score for both features and sentiments into python list
                lst_prec_exact_match_app.append(prec_exact_match_app); lst_rec_exact_match_app.append(rec_exact_match_app); lst_f1_exact_match_app.append(f1_exact_match_app)
                lst_prec_exact_match_app_sentiment.append(prec_exact_match_app_sentiment); lst_rec_exact_match_app_sentiment.append(rec_exact_match_app_sentiment); lst_f1_exact_match_app_sentiment.append(f1_exact_match_app_sentiment)

                lst_prec_partial_one_word_match_app.append(prec_partial_one_word_match_app); lst_rec_partial_one_word_match_app.append(rec_partial_one_word_match_app); lst_f1_partial_one_word_match_app.append(f1_partial_one_word_match_app)
                lst_prec_partial_one_word_match_app_sentiment.append(prec_partial_one_word_match_app_sentiment); lst_rec_partial_one_word_match_app_sentiment.append(rec_partial_one_word_match_app_sentiment); lst_f1_partial_one_word_match_app_sentiment.append(f1_partial_one_word_match_app_sentiment)

                lst_prec_partial_two_word_match_app.append(prec_partial_two_word_match_app); lst_rec_partial_two_word_match_app.append(rec_partial_two_word_match_app); lst_f1_partial_two_word_match_app.append(f1_partial_two_word_match_app)
                lst_prec_partial_two_word_match_app_sentiment.append(prec_partial_two_word_match_app_sentiment); lst_rec_partial_two_word_match_app_sentiment.append(rec_partial_two_word_match_app_sentiment); lst_f1_partial_two_word_match_app_sentiment.append(f1_partial_two_word_match_app_sentiment)


            df_prompt_eval_summary_log_data_app_wise = pd.concat([df_prompt_eval_summary_log_data_app_wise, pd.DataFrame([
                {"App_ID": app_id,'Prec_Exact_Match_Feature': "{:.1f} ± {:.1f}".format(np.mean(lst_prec_exact_match_app)*100, np.std(lst_prec_exact_match_app)*100),
                'Rec_Exact_Match_Feature':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_exact_match_app)*100, np.std(lst_rec_exact_match_app)*100),
                "F1_Exact_Match_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_exact_match_app)*100,np.std(lst_f1_exact_match_app)*100),
               'Prec_Exact_Match_Sentiment': "{:.1f} ± {:.1f}".format(np.mean(lst_prec_exact_match_app_sentiment)*100, np.std(lst_prec_exact_match_app_sentiment)*100),
               'Rec_Exact_Match_Sentiment':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_exact_match_app_sentiment)*100, np.std(lst_rec_exact_match_app_sentiment)*100),
               "F1_Exact_Match_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_exact_match_app_sentiment)*100,np.std(lst_f1_exact_match_app_sentiment)*100),
                "Prec_Partial_Match_One_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_one_word_match_app)*100, np.std(lst_prec_partial_one_word_match_app)*100),
                "Rec_Partial_Match_One_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_one_word_match_app)*100,np.std(lst_rec_partial_one_word_match_app)*100),
                "F1_Partial_Match_One_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_one_word_match_app)*100, np.std(lst_f1_partial_one_word_match_app)*100),
               "Prec_Partial_Match_One_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_one_word_match_app_sentiment)*100, np.std(lst_prec_partial_one_word_match_app_sentiment)*100),
               "Rec_Partial_Match_One_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_one_word_match_app_sentiment)*100,np.std(lst_rec_partial_one_word_match_app_sentiment)*100),
               "F1_Partial_Match_One_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_one_word_match_app_sentiment)*100, np.std(lst_f1_partial_one_word_match_app_sentiment)*100),
                "Prec_Partial_Match_Two_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_two_word_match_app)*100, np.std(lst_prec_partial_two_word_match_app)*100),
                "Rec_Partial_Match_Two_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_two_word_match_app)*100,np.std(lst_rec_partial_two_word_match_app)*100),
                "F1_Partial_Match_Two_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_two_word_match_app)*100, np.std(lst_f1_partial_two_word_match_app)*100),
               "Prec_Partial_Match_Two_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_two_word_match_app_sentiment)*100, np.std(lst_prec_partial_two_word_match_app_sentiment)*100),
               "Rec_Partial_Match_Two_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_two_word_match_app_sentiment)*100,np.std(lst_rec_partial_two_word_match_app_sentiment)*100),
               "F1_Partial_Match_Two_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_two_word_match_app_sentiment)*100, np.std(lst_f1_partial_two_word_match_app_sentiment)*100),
                "Execution time (in seconds)" :"{:.1f} ± {:.1f}".format(np.mean(lst_execution_time_app),np.std(lst_execution_time_app))}])], ignore_index=True)

            df_prompt_eval_sentiment_summary_app_wise_exact_match = pd.concat([df_prompt_eval_sentiment_summary_app_wise_exact_match, pd.DataFrame([
                {"App_ID": app_id,"Prec_Positive": "{:.1f} ± {:.1f}".format(np.mean(lst_prec_exact_match_app_pos_sentiment)*100, np.std(lst_prec_exact_match_app_pos_sentiment)*100),
                 'Rec_Positive':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_exact_match_app_pos_sentiment)*100, np.std(lst_rec_exact_match_app_pos_sentiment)*100),
                 "F1_Positive":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_exact_match_app_pos_sentiment)*100,np.std(lst_f1_exact_match_app_pos_sentiment)*100),
                 'Prec_Neutral': "{:.1f} ± {:.1f}".format(np.mean(lst_prec_exact_match_app_neutral_sentiment)*100, np.std(lst_prec_exact_match_app_neutral_sentiment)*100),
                 'Rec_Neutral':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_exact_match_app_neutral_sentiment)*100, np.std(lst_rec_exact_match_app_neutral_sentiment)*100),
                 "F1_Neutral":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_exact_match_app_neutral_sentiment)*100,np.std(lst_f1_exact_match_app_neutral_sentiment)*100),
                 "Prec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_exact_match_app_neg_sentiment)*100, np.std(lst_prec_exact_match_app_neg_sentiment)*100),
                 "Rec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_exact_match_app_neg_sentiment)*100,np.std(lst_rec_exact_match_app_neg_sentiment)*100),
                 "F1_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_exact_match_app_neg_sentiment)*100, np.std(lst_f1_exact_match_app_neg_sentiment)*100),
                 }])], ignore_index=True)

            df_prompt_eval_sentiment_summary_app_wise_partial_match_one = pd.concat([df_prompt_eval_sentiment_summary_app_wise_partial_match_one, pd.DataFrame([
                {"App_ID": app_id,"Prec_Positive": "{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_one_word_match_app_pos_sentiment)*100, np.std(lst_prec_partial_one_word_match_app_pos_sentiment)*100),
                 'Rec_Positive':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_one_word_match_app_pos_sentiment)*100, np.std(lst_rec_partial_one_word_match_app_pos_sentiment)*100),
                 "F1_Positive":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_one_word_match_app_pos_sentiment)*100,np.std(lst_f1_partial_one_word_match_app_pos_sentiment)*100),
                 'Prec_Neutral': "{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_one_word_match_app_neutral_sentiment)*100, np.std(lst_prec_partial_one_word_match_app_neutral_sentiment)*100),
                 'Rec_Neutral':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_one_word_match_app_neutral_sentiment)*100, np.std(lst_rec_partial_one_word_match_app_neutral_sentiment)*100),
                 "F1_Neutral":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_one_word_match_app_neutral_sentiment)*100,np.std(lst_f1_partial_one_word_match_app_neutral_sentiment)*100),
                 "Prec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_one_word_match_app_neg_sentiment)*100, np.std(lst_prec_partial_one_word_match_app_neg_sentiment)*100),
                 "Rec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_one_word_match_app_neg_sentiment)*100,np.std(lst_rec_partial_one_word_match_app_neg_sentiment)*100),
                 "F1_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_one_word_match_app_neg_sentiment)*100, np.std(lst_f1_partial_one_word_match_app_neg_sentiment)*100),
                 }])], ignore_index=True)

            df_prompt_eval_sentiment_summary_app_wise_partial_match_two = pd.concat([df_prompt_eval_sentiment_summary_app_wise_partial_match_two, pd.DataFrame([
                {"App_ID": app_id,"Prec_Positive": "{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_two_word_match_app_pos_sentiment)*100, np.std(lst_prec_partial_two_word_match_app_pos_sentiment)*100),
                 'Rec_Positive':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_two_word_match_app_pos_sentiment)*100, np.std(lst_rec_partial_two_word_match_app_pos_sentiment)*100),
                 "F1_Positive":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_two_word_match_app_pos_sentiment)*100,np.std(lst_f1_partial_two_word_match_app_pos_sentiment)*100),
                 'Prec_Neutral': "{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_two_word_match_app_neutral_sentiment)*100, np.std(lst_prec_partial_two_word_match_app_neutral_sentiment)*100),
                 'Rec_Neutral':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_two_word_match_app_neutral_sentiment)*100, np.std(lst_rec_partial_two_word_match_app_neutral_sentiment)*100),
                 "F1_Neutral":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_two_word_match_app_neutral_sentiment)*100,np.std(lst_f1_partial_two_word_match_app_neutral_sentiment)*100),
                 "Prec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_two_word_match_app_neg_sentiment)*100, np.std(lst_prec_partial_two_word_match_app_neg_sentiment)*100),
                 "Rec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_two_word_match_app_neg_sentiment)*100,np.std(lst_rec_partial_two_word_match_app_neg_sentiment)*100),
                 "F1_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_two_word_match_app_neg_sentiment)*100, np.std(lst_f1_partial_two_word_match_app_neg_sentiment)*100),
                 }])], ignore_index=True)


        # maintain data to compute overall performance
            lst_prec_exact_match_overall.append(np.mean(lst_prec_exact_match_app)*100);lst_rec_exact_match_overall.append(np.mean(lst_rec_exact_match_app)*100); lst_f1_exact_overall.append(np.mean(lst_f1_exact_match_app)*100)
            lst_prec_exact_match_overall_sentiment.append(np.mean(lst_prec_exact_match_app_sentiment)*100);lst_rec_exact_match_overall_sentiment.append(np.mean(lst_rec_exact_match_app_sentiment)*100); lst_f1_exact_overall_sentiment.append(np.mean(lst_f1_exact_match_app_sentiment)*100)
            lst_prec_exact_match_overall_pos_sentiment.append(np.mean(lst_prec_exact_match_app_pos_sentiment)*100);lst_rec_exact_match_overall_pos_sentiment.append(np.mean(lst_rec_exact_match_app_pos_sentiment)*100); lst_f1_exact_overall_pos_sentiment.append(np.mean(lst_f1_exact_match_app_pos_sentiment)*100)
            lst_prec_exact_match_overall_neutral_sentiment.append(np.mean(lst_prec_exact_match_app_neutral_sentiment)*100);lst_rec_exact_match_overall_neutral_sentiment.append(np.mean(lst_rec_exact_match_app_neutral_sentiment)*100); lst_f1_exact_overall_neutral_sentiment.append(np.mean(lst_f1_exact_match_app_neutral_sentiment)*100)
            lst_prec_exact_match_overall_neg_sentiment.append(np.mean(lst_prec_exact_match_app_neg_sentiment)*100);lst_rec_exact_match_overall_neg_sentiment.append(np.mean(lst_rec_exact_match_app_neg_sentiment)*100); lst_f1_exact_overall_neg_sentiment.append(np.mean(lst_f1_exact_match_app_neg_sentiment)*100)

            lst_prec_partial_one_word_match_overall.append(np.mean(lst_prec_partial_one_word_match_app)*100);lst_rec_partial_one_word_match_overall.append(np.mean(lst_rec_partial_one_word_match_app)*100);lst_f1_partial_one_word_match_overall.append(np.mean(lst_f1_partial_one_word_match_app)*100)
            lst_prec_partial_one_word_match_overall_sentiment.append(np.mean(lst_prec_partial_one_word_match_app_sentiment)*100);lst_rec_partial_one_word_match_overall_sentiment.append(np.mean(lst_rec_partial_one_word_match_app_sentiment)*100);lst_f1_partial_one_word_match_overall_sentiment.append(np.mean(lst_f1_partial_one_word_match_app_sentiment)*100)
            lst_prec_partial_one_word_match_overall_pos_sentiment.append(np.mean(lst_prec_partial_one_word_match_app_pos_sentiment)*100);lst_rec_partial_one_word_match_overall_pos_sentiment.append(np.mean(lst_rec_partial_one_word_match_app_pos_sentiment)*100);lst_f1_partial_one_word_match_overall_pos_sentiment.append(np.mean(lst_f1_partial_one_word_match_app_pos_sentiment)*100)
            lst_prec_partial_one_word_match_overall_neutral_sentiment.append(np.mean(lst_prec_partial_one_word_match_app_neutral_sentiment)*100);lst_rec_partial_one_word_match_overall_neutral_sentiment.append(np.mean(lst_rec_partial_one_word_match_app_neutral_sentiment)*100);lst_f1_partial_one_word_match_overall_neutral_sentiment.append(np.mean(lst_f1_partial_one_word_match_app_neutral_sentiment)*100)
            lst_prec_partial_one_word_match_overall_neg_sentiment.append(np.mean(lst_prec_partial_one_word_match_app_neg_sentiment)*100);lst_rec_partial_one_word_match_overall_neg_sentiment.append(np.mean(lst_rec_partial_one_word_match_app_neg_sentiment)*100);lst_f1_partial_one_word_match_overall_neg_sentiment.append(np.mean(lst_f1_partial_one_word_match_app_neg_sentiment)*100)

            lst_prec_partial_two_word_match_overall.append(np.mean(lst_prec_partial_two_word_match_app)*100);lst_rec_partial_two_word_match_overall.append(np.mean(lst_rec_partial_two_word_match_app)*100);lst_f1_partial_two_word_match_overall.append(np.mean(lst_f1_partial_two_word_match_app)*100)
            lst_prec_partial_two_word_match_overall_sentiment.append(np.mean(lst_prec_partial_two_word_match_app_sentiment)*100);lst_rec_partial_two_word_match_overall_sentiment.append(np.mean(lst_rec_partial_two_word_match_app_sentiment)*100);lst_f1_partial_two_word_match_overall_sentiment.append(np.mean(lst_f1_partial_two_word_match_app_sentiment)*100)
            lst_prec_partial_two_word_match_overall_pos_sentiment.append(np.mean(lst_prec_partial_two_word_match_app_pos_sentiment)*100);lst_rec_partial_two_word_match_overall_pos_sentiment.append(np.mean(lst_rec_partial_two_word_match_app_pos_sentiment)*100);lst_f1_partial_two_word_match_overall_pos_sentiment.append(np.mean(lst_f1_partial_two_word_match_app_pos_sentiment)*100)
            lst_prec_partial_two_word_match_overall_neutral_sentiment.append(np.mean(lst_prec_partial_two_word_match_app_neutral_sentiment)*100);lst_rec_partial_two_word_match_overall_neutral_sentiment.append(np.mean(lst_rec_partial_two_word_match_app_neutral_sentiment)*100);lst_f1_partial_two_word_match_overall_neutral_sentiment.append(np.mean(lst_f1_partial_two_word_match_app_neutral_sentiment)*100)
            lst_prec_partial_two_word_match_overall_neg_sentiment.append(np.mean(lst_prec_partial_two_word_match_app_neg_sentiment)*100);lst_rec_partial_two_word_match_overall_neg_sentiment.append(np.mean(lst_rec_partial_two_word_match_app_neg_sentiment)*100);lst_f1_partial_two_word_match_overall_neg_sentiment.append(np.mean(lst_f1_partial_two_word_match_app_neg_sentiment)*100)

            lst_execution_time_overall.append(np.mean(lst_execution_time_app))


        df_prompt_eval_overall_summary_log_data = pd.concat([df_prompt_eval_overall_summary_log_data, pd.DataFrame([{
                                                         'Prec_Exact_Match_Feature': "{:.1f} ± {:.1f}".format(np.mean(lst_prec_exact_match_overall), np.std(lst_prec_exact_match_overall)),
                                                         'Rec_Exact_Match_Feature':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_exact_match_overall), np.std(lst_rec_exact_match_overall)),
                                                         "F1_Exact_Match_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_exact_overall),np.std(lst_f1_exact_overall)),
                                                         'Prec_Exact_Match_Sentiment': "{:.1f} ± {:.1f}".format(np.mean(lst_prec_exact_match_overall_sentiment), np.std(lst_prec_exact_match_overall_sentiment)),
                                                         'Rec_Exact_Match_Sentiment':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_exact_match_overall_sentiment), np.std(lst_rec_exact_match_overall_sentiment)),
                                                         "F1_Exact_Match_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_exact_overall_sentiment),np.std(lst_f1_exact_overall_sentiment)),
                                                         "Prec_Partial_Match_One_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_one_word_match_overall), np.std(lst_prec_partial_one_word_match_overall)),
                                                         "Rec_Partial_Match_One_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_one_word_match_overall),np.std(lst_rec_partial_one_word_match_overall)),
                                                         "F1_Partial_Match_One_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_one_word_match_overall), np.std(lst_f1_partial_one_word_match_overall)),
                                                         "Prec_Partial_Match_One_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_one_word_match_overall_sentiment), np.std(lst_prec_partial_one_word_match_overall_sentiment)),
                                                         "Rec_Partial_Match_One_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_one_word_match_overall_sentiment),np.std(lst_rec_partial_one_word_match_overall_sentiment)),
                                                         "F1_Partial_Match_One_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_one_word_match_overall_sentiment), np.std(lst_f1_partial_one_word_match_overall_sentiment)),
                                                         "Prec_Partial_Match_Two_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_two_word_match_overall), np.std(lst_prec_partial_two_word_match_overall)),
                                                         "Rec_Partial_Match_Two_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_two_word_match_overall),np.std(lst_rec_partial_two_word_match_overall)),
                                                         "F1_Partial_Match_Two_Feature":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_two_word_match_overall), np.std(lst_f1_partial_two_word_match_overall)),
                                                         "Prec_Partial_Match_Two_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_two_word_match_overall_sentiment), np.std(lst_prec_partial_two_word_match_overall_sentiment)),
                                                         "Rec_Partial_Match_Two_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_two_word_match_overall_sentiment),np.std(lst_rec_partial_two_word_match_overall_sentiment)),
                                                         "F1_Partial_Match_Two_Sentiment":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_two_word_match_overall_sentiment), np.std(lst_f1_partial_two_word_match_overall_sentiment)),
                                                         "Execution time (in seconds)":"{:.1f} ± {:.1f}".format(np.mean(lst_execution_time_overall),np.std(lst_execution_time_overall))}])], ignore_index=True)

        df_prompt_eval_overall_senti_eval_summary_exact_match = pd.concat([df_prompt_eval_overall_senti_eval_summary_exact_match, pd.DataFrame([
            {"Prec_Positive": "{:.1f} ± {:.1f}".format(np.mean(lst_prec_exact_match_overall_pos_sentiment), np.std(lst_prec_exact_match_overall_pos_sentiment)),
             'Rec_Positive':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_exact_match_overall_pos_sentiment), np.std(lst_rec_exact_match_overall_pos_sentiment)),
             "F1_Positive":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_exact_overall_pos_sentiment),np.std(lst_f1_exact_overall_pos_sentiment)),
             'Prec_Neutral': "{:.1f} ± {:.1f}".format(np.mean(lst_prec_exact_match_overall_neutral_sentiment), np.std(lst_prec_exact_match_overall_neutral_sentiment)),
             'Rec_Neutral':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_exact_match_overall_neutral_sentiment), np.std(lst_rec_exact_match_overall_neutral_sentiment)),
             "F1_Neutral":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_exact_overall_neutral_sentiment),np.std(lst_f1_exact_overall_neutral_sentiment)),
             "Prec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_exact_match_overall_neg_sentiment), np.std(lst_prec_exact_match_overall_neg_sentiment)),
             "Rec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_exact_match_overall_neg_sentiment),np.std(lst_rec_exact_match_overall_neg_sentiment)),
             "F1_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_exact_overall_neg_sentiment), np.std(lst_f1_exact_overall_neg_sentiment)),
             }])], ignore_index=True)

        df_prompt_eval_overall_senti_eval_summary_partial_match_one = pd.concat([df_prompt_eval_overall_senti_eval_summary_partial_match_one, pd.DataFrame([
            {"Prec_Positive": "{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_one_word_match_overall_pos_sentiment), np.std(lst_prec_partial_one_word_match_overall_pos_sentiment)),
             'Rec_Positive':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_one_word_match_overall_pos_sentiment), np.std(lst_rec_partial_one_word_match_overall_pos_sentiment)),
             "F1_Positive":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_one_word_match_overall_pos_sentiment),np.std(lst_f1_partial_one_word_match_overall_pos_sentiment)),
             'Prec_Neutral': "{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_one_word_match_overall_neutral_sentiment), np.std(lst_prec_partial_one_word_match_overall_neutral_sentiment)),
             'Rec_Neutral':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_one_word_match_overall_neutral_sentiment), np.std(lst_rec_partial_one_word_match_overall_neutral_sentiment)),
             "F1_Neutral":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_one_word_match_overall_neutral_sentiment),np.std(lst_f1_partial_one_word_match_overall_neutral_sentiment)),
             "Prec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_one_word_match_overall_neg_sentiment), np.std(lst_prec_partial_one_word_match_overall_neg_sentiment)),
             "Rec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_one_word_match_overall_neg_sentiment),np.std(lst_rec_partial_one_word_match_overall_neg_sentiment)),
             "F1_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_one_word_match_overall_neg_sentiment), np.std(lst_f1_partial_one_word_match_overall_neg_sentiment)),
             }])], ignore_index=True)

        df_prompt_eval_overall_senti_eval_summary_partial_match_two = pd.concat([df_prompt_eval_overall_senti_eval_summary_partial_match_two, pd.DataFrame([
            {"Prec_Positive": "{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_two_word_match_overall_pos_sentiment), np.std(lst_prec_partial_two_word_match_overall_pos_sentiment)),
             'Rec_Positive':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_two_word_match_overall_pos_sentiment), np.std(lst_rec_partial_two_word_match_overall_pos_sentiment)),
             "F1_Positive":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_two_word_match_overall_pos_sentiment),np.std(lst_f1_partial_two_word_match_overall_pos_sentiment)),
             'Prec_Neutral': "{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_two_word_match_overall_neutral_sentiment), np.std(lst_prec_partial_two_word_match_overall_neutral_sentiment)),
             'Rec_Neutral':"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_two_word_match_overall_neutral_sentiment), np.std(lst_rec_partial_two_word_match_overall_neutral_sentiment)),
             "F1_Neutral":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_two_word_match_overall_neutral_sentiment),np.std(lst_f1_partial_two_word_match_overall_neutral_sentiment)),
             "Prec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_prec_partial_two_word_match_overall_neg_sentiment), np.std(lst_prec_partial_two_word_match_overall_neg_sentiment)),
             "Rec_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_rec_partial_two_word_match_overall_neg_sentiment),np.std(lst_rec_partial_two_word_match_overall_neg_sentiment)),
             "F1_Negative":"{:.1f} ± {:.1f}".format(np.mean(lst_f1_partial_two_word_match_overall_neg_sentiment), np.std(lst_f1_partial_two_word_match_overall_neg_sentiment)),
             }])], ignore_index=True)


        folder_path_class_wise_sentiment_performances_overall_exact_match = model_path_folder + "/" + "class_wise_sentiment_performances_overall_exact_match"
        folder_path_class_wise_sentiment_performances_overall_partial_match_one = model_path_folder + "/" + "class_wise_sentiment_performances_overall_partial_match_one"
        folder_path_class_wise_sentiment_performances_overall_partial_match_two = model_path_folder + "/" + "class_wise_sentiment_performances_overall_partial_match_two"

        Utility.create_folder(folder_path_class_wise_sentiment_performances_overall_exact_match)
        Utility.create_folder(folder_path_class_wise_sentiment_performances_overall_partial_match_one)
        Utility.create_folder(folder_path_class_wise_sentiment_performances_overall_partial_match_two)

        summary_log_eval_app_wise_file_name = model_path_folder + "/" + "app_wise_feature_sentiment_eval_summary_report" + ".csv"
        df_prompt_eval_summary_log_data_app_wise.to_csv(summary_log_eval_app_wise_file_name, index=False,encoding='utf-16-le')

        df_prompt_eval_sentiment_summary_app_wise_exact_match.to_csv(folder_path_class_wise_sentiment_performances_overall_exact_match + "/" + "app-wise_eval_sentiment_classes_report" + ".csv", index=False,encoding='utf-16-le')
        df_prompt_eval_sentiment_summary_app_wise_partial_match_one.to_csv(folder_path_class_wise_sentiment_performances_overall_partial_match_one + "/" + "app_wise-eval_sentiment_classes_report" + ".csv", index=False,encoding='utf-16-le')
        df_prompt_eval_sentiment_summary_app_wise_partial_match_two.to_csv(folder_path_class_wise_sentiment_performances_overall_partial_match_two + "/" + "app_wise_eval_sentiment_classes_report" + ".csv", index=False,encoding='utf-16-le')

        summary_log_eval_overall_file_name = model_path_folder + "/" +  "feature_sentiment_eval_summary_report_" + "_overall" + ".csv"
        df_prompt_eval_overall_summary_log_data.to_csv(summary_log_eval_overall_file_name, index=False,encoding='utf-16-le')
        df_prompt_eval_overall_senti_eval_summary_exact_match.to_csv(folder_path_class_wise_sentiment_performances_overall_exact_match + "/" +  "sentiment_classes_eval_report" + "_overall" + ".csv", index=False,encoding='utf-16-le')
        df_prompt_eval_overall_senti_eval_summary_partial_match_one.to_csv(folder_path_class_wise_sentiment_performances_overall_partial_match_one + "/" +  "sentiment_classes_eval_report" + "_overall" + ".csv", index=False,encoding='utf-16-le')
        df_prompt_eval_overall_senti_eval_summary_partial_match_two.to_csv(folder_path_class_wise_sentiment_performances_overall_partial_match_two + "/" +  "sentiment_classes_eval_report" + "_overall" + ".csv", index=False,encoding='utf-16-le')
        logging.info("Overall Performance summary against the prompt has been saved!!")

def main():

    parser = argparse.ArgumentParser(description='A script to evaluate the performance of LLMs for identify app features with \
                                                 their corresponding sentiments.')
    # Add positional argument
    parser.add_argument('--llm_model', type=str, default="gpt-3.5-turbo-0613", help='Name of the LLM model')
    parser.add_argument('--temperature', type=float,default=0.0,help='Set model temperature (between 0 to 1)')
    parser.add_argument('--secret_key', type=str, default="", help='API key or Token to access the model')
    parser.add_argument('--path_review_data', type=str, default='../dataset/Ground_truth.xlsx', help='Path to the labeled review dataset')
    parser.add_argument('--iterations', type=int, default=3, help='No of iterations to perform on the dataset with the LLM inference')
    parser.add_argument('--prompt_type', choices=['long', 'short'], default='short',help='Select either "long" or "short"')
    parser.add_argument('--few_shot_examples', type=int, default=0, help='Define examples to be used for few shots')
    parser.add_argument('--few_shot_examples_path', type=str, default='../prompt_data/few_shot_examples.json', help='Path to few shots examples (.csv)')
    parser.add_argument('--results_path', type=str, default='../results/', help='Path where the analysis results are generated')
    parser.add_argument('--stemmer', type=int,choices=[0,1], default=1,help='Set it to 1 to enable stemming on true and extracted features before evaluation')

    args = parser.parse_args()

    llm_model = args.llm_model
    temperature = args.temperature
    llm_secret_key = args.secret_key
    review_data_path = args.path_review_data
    no_of_iterations = args.iterations
    prompt_type = args.prompt_type
    no_few_shot_examples = args.few_shot_examples
    few_shot_example_path = args.few_shot_examples_path
    log_data_path = args.results_path
    stemmer = args.stemmer

    reviews_with_annotations_excel_file = review_data_path
    # Read the Excel file
    df_reviews_with_annotations = pd.read_excel(reviews_with_annotations_excel_file)

    # dataframe with selective columns
    df_reviews_with_features_n_sentiments = df_reviews_with_annotations[["App id","Sentence content","Feature (Positive)","Feature (Neutral)","Feature (Negative)"]]

    # Display the shapes of the resulting sets

    logging.info(f"Dataset shape: {df_reviews_with_features_n_sentiments.shape}")

    obj_feature_level_sentiment_analysis = Feature_Level_Sentiment_Analysis(llm_model,temperature,llm_secret_key,no_of_iterations,prompt_type,no_few_shot_examples,few_shot_example_path)

    obj_feature_level_sentiment_analysis.set_review_data_with_labels(df_reviews_with_features_n_sentiments)
    obj_feature_level_sentiment_analysis.set_log_data_path(log_data_path)
    obj_feature_level_sentiment_analysis.set_stemmer_value(stemmer)
    # call function for analysis (extracting features and sentiments, and do performance analysis)
    obj_feature_level_sentiment_analysis.Eval_LLMs_n_Prompts_on_Review_Data()

if __name__ == "__main__":
    main()
