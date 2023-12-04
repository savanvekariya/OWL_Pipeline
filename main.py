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


class Pipeline:
    
    def __init__(self) :
        print('Pipeline started')
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        self.tokenizer_bert = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        
        # self.nlp = spacy.load("en_core_web_sm")
        self.df = pd.DataFrame(columns = ['Sentence #','Word','Agila_DB_tag'])
        self.kb_REBEL = []
        self.preprocessRelations = None

    def getTextByCrawler(self):
        if __name__ == '__main__':
            process = CrawlerProcess()
            process.crawl(CrawlingSpider)
            process.start()

    def generateCSVFromText(self, modelName):

        with open('textOfResearchPaper.txt', '+r') as file:
            test_sentence = file.read()
        with open('bert-entity-extraction/BERT_model_tagging_task_'+modelName+'/tag_values.pkl', 'rb') as f:
            tag_values = pickle.load(f)
        with open('bert-entity-extraction/BERT_model_tagging_task_'+modelName+'/tag2idx.pkl', 'rb') as f:
            tag2idx = pickle.load(f)
        self.model_bert = BertForTokenClassification.from_pretrained('bert-entity-extraction/BERT_model_tagging_task_'+modelName)
        sentences = test_sentence.split('.')
        self.noOfSen = len(sentences)
        print(self.noOfSen)
        index = []
        word = []
        tag = []
        for idx, sentence in enumerate(sentences):
            tokenized_sentence = self.tokenizer_bert.encode(sentence)
            input_ids = torch.tensor([tokenized_sentence]).cpu()
            with torch.no_grad():
                output = self.model_bert(input_ids)
            label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
            # join bpe split tokens
            tokens = self.tokenizer_bert.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, label_indices[0]):
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(tag_values[label_idx])
                    new_tokens.append(token)
            finalResultOfTagging = {}
            for token, label in zip(new_tokens, new_labels):
                if(token != '[CLS]' and token != '[SEP]' and token.isalnum()):
                    finalResultOfTagging[token] = label
                    index.append(idx)
                    word.append(token)
                    tag.append(label)
                    # print("{}\t{}".format(label, token))
                    # new_d = {'Sentence #' : [idx], 'Word' : [token], 'Agila_DB_tag' : [label]}
                    # new_df = pd.DataFrame(new_d)
                    # self.df = pd.concat([self.df, new_df], ignore_index=True)
        
        self.df['Sentence #'] = index
        self.df['Word'] = word
        self.df['Agila_DB_tag'] = tag

    class KB:
        def __init__(self):
            self.relations = []

        def are_relations_equal(self, r1, r2):
            return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

        def exists_relation(self, r1):
            return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

        def merge_relations(self, r1):
            r2 = [r for r in self.relations if self.are_relations_equal(r1, r)][0]
            # spans_to_add = [
            #     span for span in r1["meta"]["spans"] if span not in r2["meta"]["spans"]
            # ]
            # r2["meta"]["spans"] += spans_to_add

        def add_relation(self, r):
            if not self.exists_relation(r):
                self.relations.append(r)
            else:
                self.merge_relations(r)

        def print(self):
            print("Relations:")
            for r in self.relations:
                print(f"  {r}")
        
    def extract_relations_from_model_output(self, text):
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

    def fetch_wikidata(self, query):
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

    def from_text_to_kb(self, text, span_length=128, verbose=False):
        # tokenize whole text
        inputs = self.tokenizer([text], return_tensors="pt")

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
        generated_tokens = self.model.generate(
            **inputs,
            **gen_kwargs,
        )

        # decode relations
        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=False)

        # create kb
        kb = self.KB()
        i = 0
        for sentence_pred in decoded_preds:
            current_span_index = i // num_return_sequences
            relations = self.extract_relations_from_model_output(sentence_pred)
            for relation in relations:
                
                relation["meta"] = {
                    'head':self.fetch_wikidata(relation['head']),
                    'type':self.fetch_wikidata(relation['type']),
                    'tail':self.fetch_wikidata(relation['tail'])
                    }
                kb.add_relation(relation)
            i += 1
        
        

        return kb
        
    def getTripletsFromGraphGPT(self, prompt):
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
    def get_newKB(self, kb, tags):
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

    def save_network_html(self, kb, nodes, fileName):
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

    def getKBFromGraphGPT(self, text):
        triplesData = self.getTripletsFromGraphGPT(text)
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

    def remove_punctuation(self, text):
        punctuations = string.punctuation
        text = re.sub(f"[{re.escape(punctuations)}]", " ", text)
        return re.sub("\s+", " ", text).strip()

    def getKnowledgeBase(self, kb):    
        nodes = []
        tags = dict()
        filtered_tags = dict()
        filtered_kb = []

        for item in kb:
            nodes.extend([item["head"], item["tail"]])
        nodes = list(set(nodes))

        for node in nodes:
            try:
                tag = self.df[self.df.Word == node.split(
                )[0]]["Agila_DB_tag"].value_counts().index[0]
                node = self.remove_punctuation(node)
                tag ='O'
                if(len(node)):
                    tag = self.df[self.df.Word == node.split()[0]]["Agila_DB_tag"].value_counts()
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

    def generateTriplets(self):    
        for i in range(0,self.noOfSen):    
            text = " ".join(list(self.df["Word"][self.df["Sentence #"] == i]))
            print(i, "=>",text)
            result_1 = self.from_text_to_kb(text).relations

            for item in result_1:
                # triple = dict()
                # triple['head'] = item['head']
                # triple['type'] = item['type']
                # triple['tail'] = item['tail']
                # triple['meta'] = item['meta']
                if(item not in self.kb_REBEL):
                    self.kb_REBEL.append(item)
    
    def preprocessForOWL(self):
        data = self.kb_REBEL
        
        onto = get_ontology("http://test.org/xx")
        annotations_data = pd.read_csv('bert-entity-extraction/combined_data_v8_new1.csv')
        annotations= annotations_data[['Word', 'Agila_DB_tag']]
        def Add_SubClasses(name,fullname,index,i):
            word=""
            if annotations["Agila_DB_tag"][i]=="B-"+name:
                word+=annotations["Word"][i]
                j=i+1
                
                while(annotations["Agila_DB_tag"][j]=="I-"+name):
                    word+=" "+annotations["Word"][j]
                    j+=1
                triple={
                            'head': word.strip(),
                            'type': "subclass of",
                            'tail': fullname
                        }
                if  triple not in data and word.strip() in classes:
                    data.append({
                            'head': word.strip(),
                            'type': "subclass of",
                            'tail': fullname
                        })
                    
        classes=["component","function","hardware component","hardware part","hardware subpart","measure","quantity","software","system","unit", "element"]
        #adding classes
        for el in data:
            if isinstance(el, dict):
                if el["tail"] not in classes:
                    classes.append(el["tail"])
                if el["head"] not in classes:
                    classes.append(el["head"])
            
        #adding subclasses for the super classes
        for i in range(len(annotations)):
            Add_SubClasses("comp","component",0,i)
            Add_SubClasses("func","function",1,i)
            Add_SubClasses("hwc","hardware component",2,i)
            Add_SubClasses("hwp","hardware part",3,i)
            Add_SubClasses("hwsp","hardware subpart",4,i)
            Add_SubClasses("mea","measure",5,i)
            Add_SubClasses("qt","quantity",6,i)
            Add_SubClasses("sw","software element",7,i)
            Add_SubClasses("sys","system",8,i)
            Add_SubClasses("unit","unit",9,i)
            Add_SubClasses("elem","element",10,i)
            Add_SubClasses("hw","hardware",10,i)

        #lowercase change
        for el in data:
            el["head"]=el["head"].lower()
            el["type"]=el["type"].lower()
            el["tail"]=el["tail"].lower()
        #print(data[-100:-1])
        #saving data in pickle file
        new_data = []
        for item in data:
            if((not re.match(r"\W", item['head'])) and (not re.match(r"\W", item['tail']))):
                triple = dict()
                triple['head'] = item['head']
                triple['type'] = item['type']
                triple['tail'] = item['tail']
                new_data.append(triple)
        self.preprocessRelations = new_data

    def createOWLFile(self, name):
        data = self.preprocessRelations
        print(len(data))
        onto = get_ontology("GENIALOntBFO.owl").load()
        #list of the 10 tags
        all_classes=list(onto.classes())
        object_properties = list(onto.object_properties())
        total1=[]
        total2=[]
        total3=[]
        total4=[]
        total5=[]
        total6=[]
        total7=[]
        functionsnamespaces=[]
        for el in all_classes:
            length=max(el.iri.rfind("/"),el.iri.rfind("#"))+1
            if el.iri[0:length]=="http://w3id.org/gbo#":
                total1.append(el.iri[length:].replace("_"," "))
            if el.iri[0:length]=="http://w3id.org/gbo/CarModel/":
                total2.append(el.iri[length:].replace("_"," "))
            if el.iri[0:length]=="http://www.ontology-of-units-of-measure.org/resource/om-2/":
                total3.append(el.iri[length:].replace("_"," "))
            if el.iri[0:length]=="http://w3id.org/gbo/":
                total4.append(el.iri[length:].replace("_"," "))
            if el.iri[0:length]=="http://www.w3.org/ns/sosa/":
                total5.append(el.iri[length:].replace("_"," "))
        for el in object_properties:
            length=max(el.iri.rfind("/"),el.iri.rfind("#"))+1
            if el.iri[0:length]=="http://w3id.org/gbo#":
                total6.append(el.iri[length:].replace("_"," "))   
            if el.iri[0:length]=="http://www.ontology-of-units-of-measure.org/resource/om-2/":
                total7.append(el.iri[length:].replace("_"," "))
            
        with onto:
            global a1, a2, a3, a4, a5
            a1 = get_namespace("http://w3id.org/gbo#")
            a2 = get_namespace("http://w3id.org/gbo/CarModel/")
            a3 = get_namespace("http://www.ontology-of-units-of-measure.org/resource/om-2/")
            a4 = get_namespace("http://w3id.org/gbo/")
            a5 = get_namespace("http://www.w3.org/ns/sosa/")
        #fixing direct graph cycles
        for i in range(len(data)):
            if data[i]["type"]=="subclass of" :
                for i1 in range(i+1,len(data)):
                    if data[i1]["type"]=="subclass of" :
                        if data[i]["head"]==data[i1]["tail"] and data[i]["tail"]==data[i1]["head"]:
                            data[i]="0"
                            break
        for el in data:
            if el=="0":
                data.remove(el)
            else:
                if el["head"]==el["tail"] and el["type"]=="subclass of":
                    data.remove(el)
        #deleting instances where one of head or tail is ""
                if (el["head"]=="" or el ["tail"]==""):
                    data.remove(el)
        #deleting subclass that defies the Disjoint Rule
                if el["type"]=="subclass of" and el["head"] in ["hardware component","hardware elementary subpart","hardware part","hardware subpart"]  and el["tail"] in ["hardware component","hardware elementary subpart","hardware part","hardware subpart"]:
                    data.remove(el)
        #Function to save Ontology in an owl file function
        def save():
            onto.save(file = name+".owl", format = "rdfxml")
        #function to code strings / to change unallowed symbols
        def change(el:str):
            if el in total1 or el in total2 or el in total3 or el in total4 or el in total5 or el in total6 or el in total7:
                return(el.replace(" ","_"))
            return "_"+el.replace("�", "A0").replace("–", "A1").replace(":", "A2").replace(" ", "_").replace("$", "A3").replace("(", "A4").replace(")", "A5").replace(",", "A6").replace("−", "A7").replace("+", "A8").replace("'", "A9").replace("/", "A10").replace(".", "A11").replace("-", "A12").replace("²", "A13").replace("¼", "A14").replace("½", "A15")
            #return el.replace("�", "A0").replace("–", "A1").replace(":", "A2").replace(" ", "_").replace("$", "A3").replace("(", "A4").replace(")", "A5").replace(",", "A6").replace("−", "A7").replace("+", "A8").replace("'", "A9").replace("/", "A10").replace(".", "A11").replace("-", "A12").replace("²", "A13").replace("¼", "A14").replace("½", "A15")
            
        #onto.save(file = "test.owl", format = "rdfxml")
        classes=[]
        data1=data
        #print(data[0:50])
        #adding classes
        for el in data:
            if isinstance(el, dict):
                if el["tail"] not in classes:
                    classes.append(el["tail"])
                if el["head"] not in classes:
                    classes.append(el["head"])
            
        #Defining subcalsses List(each class of index i has his subclasses at index i)
        SubClasses=[ [] for i in range(len(classes))]
        instanceof=[]#list of dictionnaries with triples and type=="instance of"
        instances=[]#list of instances
        for el in data:
        #adding subclasses for el in data:
            if(type(el) == dict):
                if el["type"]=="subclass of":
                    index = classes.index(el["head"])
                    if el["tail"] not in SubClasses[index]:
                        SubClasses[index].append(el["tail"])
        #adding relations to the instanceS list
        for el in data:
            if(type(el) == dict):
                if el["type"]=="instance of":
                    instanceof.append(el)
                    instances.append(el["head"])
        #weird bug manual correction
        for i in range(len(SubClasses)):
            if classes[i]=="workflow" and SubClasses[i]==["workflow"]:
                SubClasses[i]=[]
        #adding classes to Ontology         
        for i in range(len(SubClasses)):
            if SubClasses[i]==[]:
                if classes[i] not in total1 and classes[i] not in total2 and classes[i] not in total3 and classes[i] not in total4 and classes[i] not in total5:
                    class_name=change(classes[i])
                    label=classes[i]
                    exec(f"with onto: \n  class {class_name}(Thing): \n   label=\"{label}\" ")
            else:
                subclasses_of_el=""
                class_name=change(classes[i])
                label=classes[i]
                for subclass in SubClasses[i]:
                    onto_subclass="onto"
                    if subclass in total1:
                        onto_subclass="a1"
                    if subclass in total2:
                        onto_subclass="a2"
                    if subclass in total3:
                        onto_subclass="a3"
                    if subclass in total4:
                        onto_subclass="a4"
                    if subclass in total5:
                        onto_subclass="a5"
                    if subclasses_of_el=="":
                        subclasses_of_el+=onto_subclass+"."+change(subclass)
                    else:
                        subclasses_of_el+=","+onto_subclass+"."+change(subclass)  # CPU,Compenent
                    if subclass not in total1:
                        missingclass=change(subclass)
                        exec(f"with onto: \n  if not onto.{missingclass}: \n    class {missingclass}(Thing): \n      pass")
                if classes[i] not in total1 and classes[i] not in total2 and classes[i] not in total3 and classes[i] not in total4 and classes[i] not in total5:
                    print(f"with onto: \n  class {class_name}({subclasses_of_el}): \n    label=\"{label}\" ")
                    exec(f"with onto: \n  class {class_name}({subclasses_of_el}): \n    label=\"{label}\" ")

                else:
                    if classes[i] in total1:
                        exec(f"with onto: \n  class {class_name}({subclasses_of_el}): \n    label=\"{label}\" \n    namespace=a1 ")
                    if classes[i] in total2:
                        exec(f"with onto: \n  class {class_name}({subclasses_of_el}): \n    label=\"{label}\" \n    namespace=a2 ")
                    if classes[i] in total3:
                        exec(f"with onto: \n  class {class_name}({subclasses_of_el}): \n    label=\"{label}\" \n    namespace=a3 ")
                    if classes[i] in total4:
                        exec(f"with onto: \n  class {class_name}({subclasses_of_el}): \n    label=\"{label}\" \n    namespace=a4 ")
                    if classes[i] in total5:
                        exec(f"with onto: \n  class {class_name}({subclasses_of_el}): \n    label=\"{label}\" \n    namespace=a5 ")
        #adding instances to the ontology
        for el in instanceof:
            onto_tail="onto"
            onto_head="onto"
            if el["head"]  in total1:
                onto_head="a1"
            if el["head"]  in total2:
                onto_head="a2"
            if el["head"]  in total3:
                onto_head="a3"
            if el["head"]  in total4:
                onto_head="a4"
            if el["head"]  in total5:
                onto_head="a5"
            if el["tail"]  in total1:
                onto_tail="a1"
            if el["tail"]  in total2:
                onto_tail="a2"
            if el["tail"]  in total3:
                onto_tail="a3"
            if el["tail"]  in total4:
                onto_tail="a4"
            if el["tail"]  in total5:
                onto_tail="a5"
            head=el["head"]+"_"
            label=el["head"]
            if head in total1 or head in total2 or head in total3 or head in total4 or head in total5:
                head="_"+el["head"]
            tail=change(el["tail"])
            changed_head=change(el["head"])
            #print(head+" "+tail)
            exec(f"with onto: \n  instance={onto_tail}.{tail}(\"{head}\") \n  instance.label=\"{label}\"")
            exec(f"with onto: \n  instance={onto_head}.{changed_head}(\"{head}\")")
        #adding relations to the ontology
        for el in data1:
            if(type(el)==dict):
                onto_rel="onto"
                if el["type"] in total7:
                    onto_rel="a3"
                if el["type"] in total6:
                    onto_rel="a1"
                if el["type"] in total6 or el["type"] in total7:
                    relation=change(el["type"])
                else:
                    relation="_"+change(el["type"])
                if el["type"]=="instance of":
                    continue
                if el["type"]!="subclass of":
                    onto_tail="onto"
                    onto_head="onto"
                    if el["head"]  in total1:
                        onto_head="a1"
                    if el["head"]  in total2:
                        onto_head="a2"
                    if el["head"]  in total3:
                        onto_head="a3"
                    if el["head"]  in total4:
                        onto_head="a4"
                    if el["head"]  in total5:
                        onto_head="a5"
                    if el["tail"]  in total1:
                        onto_tail="a1"
                    if el["tail"]  in total2:
                        onto_tail="a2"
                    if el["tail"]  in total3:
                        onto_tail="a3"
                    if el["tail"]  in total4:
                        onto_tail="a4"
                    if el["tail"]  in total5:
                        onto_tail="a5"
                    label=el["type"]
                    head=change(el["head"])
                    tail=change(el["tail"])
                    head_unchanged=(el["head"])
                    tail_unchanged=(el["tail"])
                    #print(label+" "+head+" "+tail )
                    if el["type"] not in total6 and el["type"] not in total7:
                        exec(f"with onto: \n  class {relation}(ObjectProperty): \n    label=\"{label}\" ")
                    if head_unchanged in instances and tail_unchanged in instances:
                        exec(f"with onto: \n  a=onto.search(iri = \"*http://w3id.org/gbo/mgg#{head_unchanged}*\")[0] \n  b=onto.search(iri = \"*http://w3id.org/gbo/mgg#{tail_unchanged}*\")[0] \n  a.{relation}.append(b)")
                        continue        
                    exec(f"{onto_head}.{head}.is_a.append({onto_rel}.{relation}.some({onto_tail}.{tail}))")

            #changing class names/removing _ or __ and adding function
            all_classes=list(onto.classes())
            object_properties = list(onto.object_properties())
            # for el in all_classes:
            #     index =max(el.iri.rfind("/"),el.iri.rfind("#"))+1
            #     if el.iri[index]=="_" :
            #         print(el.iri)
            #         el.iri = el.iri[:index] + el.iri[index+1:]
            # for el in object_properties:
            #     index =max(el.iri.rfind("/"),el.iri.rfind("#"))+1
            #     if el.iri[index]=="_" :
            #         if el.iri[index+1]=="_" :
            #             el.iri = el.iri[:index] + el.iri[index+2:]+"_function"
            #             continue
            #         el.iri = el.iri[:index] + el.iri[index+1:]+"_function"
                    
            save()
            #exec(f"with onto: \n  class {class_name}(Thing): pass")
            #with onto:
            #    class Drug(Thing):
            #         pass


    def start(self, modelClassType,owlFileName):
        self.getTextByCrawler()
        self.generateCSVFromText(modelClassType)
        self.generateTriplets()
        self.preprocessForOWL()
        self.createOWLFile(owlFileName)

c = Pipeline()
c.start('normal', 'es_normal')