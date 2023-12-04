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
class Rebel:
    class KB:
        def __init__(self):
            self.relations = []

        def are_relations_equal(self, r1, r2):
            return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

        def exists_relation(self, r1):
            return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

        def merge_relations(self, r1):
            r2 = [r for r in self.relations if self.are_relations_equal(r1, r)][0]
            spans_to_add = [
                span for span in r1["meta"]["spans"] if span not in r2["meta"]["spans"]
            ]
            r2["meta"]["spans"] += spans_to_add

        def add_relation(self, r):
            if not self.exists_relation(r):
                self.relations.append(r)
            else:
                self.merge_relations(r)

        def print(self):
            print("Relations:")
            for r in self.relations:
                print(f"  {r}")
        
    def extract_relations_from_model_output(text):
        relations = []
        relation, subject, relation, object_ = "", "", "", ""
        text = text.strip()
        current = "x"
        text_replaced = text.replace("<s>", "").replace(
            "<pad>", "").replace("</s>", "")
        for token in text_replaced.split():
            if token == "<triplet>":
                current = "t"
                if relation != "":
                    relations.append(
                        {
                            "head": subject.strip(),
                            "type": relation.strip(),
                            "tail": object_.strip(),
                        }
                    )
                    relation = ""
                subject = ""
            elif token == "<subj>":
                current = "s"
                if relation != "":
                    relations.append(
                        {
                            "head": subject.strip(),
                            "type": relation.strip(),
                            "tail": object_.strip(),
                        }
                    )
                object_ = ""
            elif token == "<obj>":
                current = "o"
                relation = ""
            else:
                if current == "t":
                    subject += " " + token
                elif current == "s":
                    object_ += " " + token
                elif current == "o":
                    relation += " " + token
        if subject != "" and relation != "" and object_ != "":
            relations.append(
                {"head": subject.strip(), "type": relation.strip(),
                "tail": object_.strip()}
            )
        return relations

    def fetch_wikidata(query):
        url = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'search': query,
            'language': 'en'
        }
        try:
            return requests.get(url, params=params).json()['search'][0]['id']
        except:
            return 'There was and error'

    def from_text_to_kb(text, span_length=128, verbose=False):
        # tokenize whole text
        inputs = tokenizer([text], return_tensors="pt")

        # compute span boundaries
        num_tokens = len(inputs["input_ids"][0])
        if verbose:
            print(f"Input has {num_tokens} tokens")
        num_spans = math.ceil(num_tokens / span_length)
        if verbose:
            print(f"Input has {num_spans} spans")
        overlap = math.ceil(
            (num_spans * span_length - num_tokens) / max(num_spans - 1, 1))
        spans_boundaries = []
        start = 0
        for i in range(num_spans):
            spans_boundaries.append(
                [start + span_length * i, start + span_length * (i + 1)]
            )
            start -= overlap
        if verbose:
            print(f"Span boundaries are {spans_boundaries}")

        # transform input with spans
        tensor_ids = [
            inputs["input_ids"][0][boundary[0]: boundary[1]]
            for boundary in spans_boundaries
        ]
        tensor_masks = [
            inputs["attention_mask"][0][boundary[0]: boundary[1]]
            for boundary in spans_boundaries
        ]
        inputs = {
            "input_ids": torch.stack(tensor_ids),
            "attention_mask": torch.stack(tensor_masks),
        }


        # generate relations
        num_return_sequences = 3
        gen_kwargs = {
            "max_length": 256,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": num_return_sequences,
        }
        generated_tokens = model.generate(
            **inputs,
            **gen_kwargs,
        )

        # decode relations
        decoded_preds = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=False)

        # create kb
        kb = KB()
        i = 0
        for sentence_pred in decoded_preds:
            current_span_index = i // num_return_sequences
            relations = extract_relations_from_model_output(sentence_pred)
            for relation in relations:
                
                relation["meta"] = {
                    'head':fetch_wikidata(relation['head']),
                    'type':fetch_wikidata(relation['type']),
                    'tail':fetch_wikidata(relation['tail'])
                    }
                kb.add_relation(relation)
            i += 1
        
        

        return kb

    #tags => PERSON, EVENT, LOC, PRODUCT, ORG

    def check_additional_tags(text, tag, owl_class):
        sentence = Sentence(text)
        tagger.predict(sentence)
        # print(sentence.get_spans('ner'))
        relation_from_rebel = []
        for entity in sentence.get_spans('ner'):
            if(entity.tag == tag):
                relation_from_rebel = from_text_to_kb(text).relations
                relation_from_rebel.append({'head': entity.text, 'type':'subclass of' , 'tail': owl_class})
                # print(entity.text, entity.tag, entity.score)

        return relation_from_rebel

    def check_additional_tags_bert_ner(text, tag , owl_class):
        subclass_relations = []
        nlp = pipeline("ner", model=model3, tokenizer=tokenizer3, grouped_entities=True)
        ner_results = nlp(text)
        entity_List = [item['entity_group'] for item in ner_results]

        for item in ner_results:
            if(item['entity_group'] == tag):
                subclass_relations.append({'head': item['word'], 'type':'subclass of' , 'tail': owl_class})
                
        if(tag in entity_List):
            relations_from_rebel = from_text_to_kb(text).relations
            relations_from_rebel = [*relations_from_rebel, *subclass_relations]
            return relations_from_rebel
        return []
        
    def check_additional_tags_spacy_transformers(text, tag, owl_class):
        roberta_nlp = spacy.load("en_core_web_sm")
        document = roberta_nlp(text)
        subclass_relations = []
        has_tag = False
        for entity in document.ents:
                # print(entity.text + '->', entity.label_)
                if(entity.label_ in tag):
                    has_tag = True
                    triplet = {'head': entity.text, 'type':'subclass of' , 'tail': owl_class}
                    if(triplet not in subclass_relations):
                        subclass_relations.append(triplet)
        if(has_tag):
            relations_from_rebel = from_text_to_kb(text).relations
            relations_from_rebel = [*relations_from_rebel, *subclass_relations]
            return relations_from_rebel
        return []

    def getTripletsFromGraphGPT(prompt):
        # Load the prompt from the file
        with open('prompts/stateless.prompt', 'r') as f:
            text = f.read()
        # Replace the $prompt variable in the text with the actual prompt
        text = text.replace('$prompt', prompt)

        # Create the request parameters
        default_params = {
        "model": "text-davinci-003",
        "temperature": 0.3,
        "max_tokens": 800,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
        }
        params = {**default_params, 'prompt': text, 'stop': '\n'}
        api_key='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'
        # Create the request options
        url = 'https://api.openai.com/v1/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        data = json.dumps(params)

        # Send the request
        response = requests.post(url, headers=headers, data=data)

        # Check the response status
        if response.ok:
            # Parse the response data
            response_json = response.json()
            choices = response_json['choices']
            text = choices[0]['text']
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

    # forming relations between the nodes
    def get_newKB(kb, tags):
        new_kb = []
        for dp in kb:
            d_ = dict()
            d_["head"] = dp["head"]
            d_["tail"] = dp["tail"]

            if tags[dp["head"]] == "sys" and tags[dp["tail"]] == "comp":
                d_["type"] = "has part directly"
                if d_ not in new_kb:
                    new_kb.append(d_)
            elif tags[dp["head"]] == "comp" and tags[dp["tail"]] == "sys":
                d_["type"] = "direct part of"
                if d_ not in new_kb:
                    new_kb.append(d_)

            elif tags[dp["head"]] == "hwc" and tags[dp["tail"]] == "hwp":
                d_["type"] = "has part directly"
                if d_ not in new_kb:
                    new_kb.append(d_)
            elif tags[dp["head"]] == "hwp" and tags[dp["tail"]] == "hwc":
                d_["type"] = "direct part of"
                if d_ not in new_kb:
                    new_kb.append(d_)

            elif (
                tags[dp["head"]] not in ["mea", "qt", "unit", "func"]
                and tags[dp["tail"]] == "func"
            ):
                d_["type"] = "implements"
                if d_ not in new_kb:
                    new_kb.append(d_)
            elif tags[dp["head"]] == "func" and tags[dp["tail"]] not in [
                "mea",
                "qt",
                "unit",
            ]:
                d_["type"] = "implemented by"
                if d_ not in new_kb:
                    new_kb.append(d_)

            elif (
                tags[dp["head"]] in ["sys", "comp", "hwc", "hwp", "hwsp"]
                and tags[dp["tail"]] == "sw"
            ):
                d_["type"] = "executes"
                if d_ not in new_kb:
                    new_kb.append(d_)
            elif tags[dp["head"]] == "sw" and tags[dp["tail"]] in [
                "sys",
                "comp",
                "hwc",
                "hwp",
                "hwsp",
            ]:
                d_["type"] = "executed by"
                if d_ not in new_kb:
                    new_kb.append(d_)

            elif tags[dp["head"]] == "hwsp" and tags[dp["tail"]] == "hwp":
                d_["type"] = "part of directly"
                if d_ not in new_kb:
                    new_kb.append(d_)
            elif tags[dp["head"]] == "hwp" and tags[dp["tail"]] == "hwsp":
                d_["type"] = "has part"
                if d_ not in new_kb:
                    new_kb.append(d_)

            elif (
                tags[dp["head"]] not in ["mea", "func", "unit", "qt"]
                and tags[dp["tail"]] == "qt"
            ):
                d_["type"] = "has property"
                if d_ not in new_kb:
                    new_kb.append(d_)
            elif tags[dp["head"]] == "func" and tags[dp["tail"]] not in [
                "mea",
                "func",
                "unit",
                "qt",
            ]:
                d_["type"] = "property of"
                if d_ not in new_kb:
                    new_kb.append(d_)

            elif tags[dp["head"]] == "qt" and tags[dp["tail"]] == "mea":
                d_["type"] = "has value"
                if d_ not in new_kb:
                    new_kb.append(d_)
            elif tags[dp["head"]] == "mea" and tags[dp["tail"]] == "qt":
                d_["type"] = "value of"
                if d_ not in new_kb:
                    new_kb.append(d_)

            elif tags[dp["head"]] == "mea" and tags[dp["tail"]] == "unit":
                d_["type"] = "has unit"
                if d_ not in new_kb:
                    new_kb.append(d_)
            elif tags[dp["head"]] == "unit" and tags[dp["tail"]] == "mea":
                d_["type"] = "unit of"
                if d_ not in new_kb:
                    new_kb.append(d_)

        return new_kb

    def save_network_html(kb, nodes, fileName):
        # create network
        net = Network(
            directed=True, width="1500px", height="900px", bgcolor="#eeeeee", notebook=True
        )

        # nodes
        color_entity = "#00FF00"
        for e in nodes:
            net.add_node(e, shape="circle", color=color_entity)

        # edges
        for r in kb:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])

        # save network
        net.repulsion(
            node_distance=200,
            central_gravity=0.2,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09,
        )
        net.set_edge_smooth("dynamic")
        net.show(fileName)

    def getKBFromGraphGPT(text):
        triplesData = getTripletsFromGraphGPT(text)
        triples = eval(triplesData)
        graphGPTKB = []
        for triple in triples:
            if(len(triple)>=3):
                graphGPTKB.append({
                    'head':triple[0],
                    'type':triple[1],
                    'tail':triple[2]
                })
        return graphGPTKB

    def remove_punctuation(text):
        punctuations = string.punctuation
        text = re.sub(f"[{re.escape(punctuations)}]", " ", text)
        return re.sub("\s+", " ", text).strip()

    def getKnowledgeBase(kb):    
        nodes = []
        tags = dict()
        filtered_tags = dict()
        filtered_kb = []

        for item in kb:
            nodes.extend([item["head"], item["tail"]])
        nodes = list(set(nodes))

        for node in nodes:
            try:
                tag = df[df.Word == node.split(
                )[0]]["Agila_DB_tag"].value_counts().index[0]
                node = remove_punctuation(node)
                tag ='O'
                if(len(node)):
                    tag = df[df.Word == node.split()[0]]["Agila_DB_tag"].value_counts()
                    if(tag.count()):
                        if('O' in tag):
                            tag = tag.drop('O')
                        if(tag.count()):
                            tag = tag.index[0]
                        else:
                            tag = 'O' 
                    else:
                        tag = 'O'
                else:
                    tag ='O'
                tags[node] = tag
            except:
                pass

        for key, value in tags.items():
            if value != "O":
                filtered_tags[key] = value.split("-")[-1]

        tag_nodes = list(filtered_tags.keys())

        for dp in kb:
            f_kb = dict()
            if (dp["head"] in tag_nodes) and (dp["tail"] in tag_nodes):
                if dp["head"] != dp["tail"]:
                    f_kb["head"] = dp["head"]
                    f_kb["type"] = dp["type"]
                    f_kb["tail"] = dp["tail"]
                    filtered_kb.append(f_kb)

        new_kb = get_newKB(filtered_kb, filtered_tags)

        nodes = []
        for r in new_kb:
            nodes.extend([r["head"], r["tail"]])
        
        return {'kb': new_kb, 'nodes' : nodes}
