import  math, torch, IPython, requests, json, pickle, pandas as pd, re, string, spacy, numpy as np, csv, scrapy, json
from scrapy.crawler import CrawlerProcess
from owlready2 import *
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizer,
    BertForTokenClassification
    )
from crawler import CrawlingSpider
from pyvis.network import Network
import openai
import nltk
import time


class GPTPipeline:

    def __init__(self):
        self.test_sentence = ""
        
    def getTripletsFromGraphGPT(self, prompt):
        # Load the prompt from the file
        with open('prompts/custom.prompt', 'r') as f:
            text = f.read()
        # Replace the $prompt variable in the text with the actual prompt
        text = text.replace('$prompt', prompt)
        print(text)
        # Create the request parameters
        default_params = {
        "model": "text-davinci-003",
        "temperature": 0.3,
        "max_tokens": 2000,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
        }
        params = {**default_params, 'prompt': text, 'stop': '\n'}
        api_key='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'
        # api_key='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'
        # api_key='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'
        # Create the request options
        url = 'https://api.openai.com/v1/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        data = json.dumps(params)

        # Send the request
        response = requests.post(url, headers=headers, data=data)
        print(response)
        # Check the response status
        if response.ok:
            # Parse the response data
            response_json = response.json()
            choices = response_json['choices']
            print(choices)
            text = choices[0]['text']
            print(text)
            # updates = json.loads(text)
            return text
        else:
            # Handle the error
            status_code = response.status_code
            if status_code == 401:
                raise Exception('Please double-check your API key.')
            elif status_code == 429:
                raise Exception('You exceeded your current quota, please check your plan and billing details.')
            else:
                raise Exception('Something went wrong with the request, please check the Network log')


    def gpt4(self, prompt):
        with open('prompts/custom.prompt', 'r') as f:
            text = f.read()
        # Replace the $prompt variable in the text with the actual prompt
        text = text.replace('$prompt', prompt)
        openai.api_key ='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'
        messages = [
            {"role":"system", "content":"You are knowledge graph expert."},
            {"role":"user", "content":text}
        ]
        ans = openai.ChatCompletion.create(
            model="gpt-4",
            max_tokens=2048,
            messages=messages
        )
        time.sleep(10)
        tripletStr = ans['choices'][0]['message']['content']
        pattern = "[^0-9a-zA-Z:\"\[\]{},\s]+"
        clean_string = re.sub(pattern, "", tripletStr)
        triplet = eval(clean_string)
        return triplet

    def getKBFromGraphGPT(self, text):
        triplesData = self.getTripletsFromGraphGPT(text)
        print(triplesData)
        # triples = eval(triplesData)
        # graphGPTKB = []
        # for triple in triples:
        #     if(len(triple)>=3):
        #         graphGPTKB.append({
        #             'head':triple[0],
        #             'type':triple[1],
        #             'tail':triple[2]
        #         })
        # return graphGPTKB
    
    def tripletsFromGPT(self):
        with open('textOfResearchPaper.txt', '+r') as file:
            self.test_sentence = file.read()
        sentences = self.test_sentence.replace(',', "").replace("[", ""). replace("]", ""). replace('\n', "").split('.')
        resultArr = []
        for text in sentences:
            pattern = "[^0-9a-zA-Z:,\s]+"
            clean_string = re.sub(pattern, "", text)
            if (len(clean_string) > 50):
                arrTriplet = self.gpt4(clean_string)
                # resultArr.extend(arrTriplet)
                print(arrTriplet)
        
        


    def start(self):
        self.tripletsFromGPT()
        # self.getTextByCrawler()
        # self.generateCSVFromText()
        # self.generateTriplets()
        # self.preprocessForOWL()
        # self.createOWLFile()
        # print(self.df.tail(10))
        # print(self.kb_REBEL)
        # print(self.preprocessRelations)

c = GPTPipeline()
c.start()