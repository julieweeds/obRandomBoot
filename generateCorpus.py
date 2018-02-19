import pandas as pd
import matplotlib as plt
from collections import defaultdict
import random
import nlp_tools
import spacy,operator
import logging
nlp=spacy.load('en')

def make_countdict(alldata):
    countdict = {}
    blacklist = ['words', 'obc_hiscoCode']

    for heading in alldata.columns:
        # print('Generating counts for ' +heading)
        if heading not in blacklist:
            countdict[heading] = defaultdict(int)
            selection = alldata[heading]
            for item in selection:
                # print(item)
                countdict[heading][item] += 1
        else:
            # print('skipping')
            pass

    return countdict


def validated(reqlist, valuedata):
    reqdict = {}
    for (field, value) in reqlist:

        parts = field.split(':')
        if len(parts) == 1:
            if field in valuedata.keys():
                if isinstance(value, list):
                    ok = []
                    for v in value:
                        if v in valuedata[field].keys():
                            ok.append(v)
                    if len(ok) > 0:
                        reqdict[field] = ok
                elif value in valuedata[field].keys():
                    reqdict[field] = value


        else:
            if parts[1]=="not" and parts[0] in valuedata.keys():
                if value in valuedata[parts[0]].keys():
                    reqdict[field] = value

            elif (parts[1] == "max" or parts[1] == "min") and parts[0] in valuedata.keys():

                if isinstance(value, list):
                    logging.warning("Error: not, min and max cannot be list")

                elif value in valuedata[parts[0]].keys() and isinstance(value, int):
                    reqdict[field] = value

    return reqdict


def find_trials(worddf, trialdf,reqlist, join='obo_trial'):
    trialreqdict = validated(reqlist, make_countdict(trialdf))
    wordsreqdict = validated(reqlist, make_countdict(worddf))

    logging.info(trialreqdict)
    logging.info(wordsreqdict)
    ok = True
    for (req, _value) in reqlist:
        if req in trialreqdict.keys() or req in wordsreqdict.keys():
            pass
        else:
            logging.warning("Requirement {} not satisfied".format(req))
            ok = False

    if not ok:
        return None

    trials = trialdf
    for req in trialreqdict.keys():
        parts = req.split(':')
        value = trialreqdict[req]
        if len(parts)>1:
            if parts[1]=='not':
                trials=trials[trials[parts[0]]!=value]
            elif parts[1]=='max':
                trials=trials[trials[parts[0]]<=value]
            elif parts[1]=='min':
                trials=trials[trials[parts[0]]>=value]
        elif isinstance(value,list):
            trials=trials[trials[req].isin(value)]
        else:
            trials=trials[trials[req]==value]


    selection=[line for line in trials[join]]
    return selection

def name(reqs):
    output_file=''
    for (field,characteristic) in reqs:
        
        parts=field.split(':')
        if len(parts)>1:
            affix=parts[1]+":"
        else:
            affix=''
        
        if isinstance(characteristic,list):
            for c in characteristic:
                output_file+="_"+affix+str(c)
        else:
            output_file+="_"+affix+str(characteristic)
    return output_file

def build_corpus(wdf, tdf, reqs):
    
    output_file="corpus"+name(reqs)
    

    print("Writing to {}".format(output_file))        
    trials = find_trials(wdf, tdf, reqs)
    # print(len(trials),trials)
    c = generate_corpus(wdf, trials, reqs)
    

    with open(output_file,"w") as output:
    
        for item in c:
            output.write(item+"\n")
    
    


def generate_corpus(worddata, trials, reqs,prop=100):
    N = len(trials)
    corpus = []
    triallabels=[]
    N=int(N*prop/100)
    allreqdict = validated(reqs, make_countdict(worddata))
    for i in range(0, N):
        atrial = trials[i]
        wdf = worddata[worddata['obo_trial'] == atrial]
        for req in allreqdict.keys():
            parts = req.split(':')
            value = allreqdict[req]
            if len(parts) > 1:
                if parts[1]=='not':
                    wdf=wdf[wdf[parts[0]]!=value]
                elif parts[1] == 'max':
                    wdf = wdf[wdf[parts[0]] <= value]
                elif parts[1] == 'min':
                    wdf = wdf[wdf[parts[0]] >= value]
            elif isinstance(value, list):
                wdf = wdf[wdf[req].isin(value)]
            else:
                wdf = wdf[wdf[req] == value]

        corpus += [(line,atrial) for line in wdf['words']]
    return corpus

def random_split(wdf,tdf,reqs,prop=50,cache=False,testing=False,seed=5):
    rev=100-prop
    outfile1="random_"+str(seed)+"A_"+str(prop)+"_"+name(reqs)
    outfile2="random_"+str(seed)+"B_"+str(rev)+"_"+name(reqs)
    
    if testing:
        loadprop=10
    else:
        loadprop=100
    if cache==True:
        try:
            c1=nlp_tools.corpus([outfile1],nlp,prop=loadprop,ner=False,paired=True)
            c2=nlp_tools.corpus([outfile2],nlp,prop=loadprop,ner=False,paired=True)
            return(c1,c2)    
              
        except:
            
            pass
    
    trials = find_trials(wdf, tdf, reqs)
    #print(len(trials),trials)
    #print(len(trials))
    c = generate_corpus(wdf, trials, reqs)
    #print(len(c))
    random.seed(seed)
    random.shuffle(c)
    cutoff=int((prop/100)*len(c))
    
    with open(outfile1,"w") as output:
    
        for (item,label) in c[:cutoff]:
            output.write(unicode(item+"\t"+label+"\n",encoding="UTF-8"))
    with open(outfile2,"w") as output:
        for (item,label) in c[cutoff:]:
            output.write(unicode(item+"\t"+label+"\n",encoding="UTF-8"))
    c1=nlp_tools.corpus([outfile1],nlp,prop=loadprop,ner=False,paired=True)
    c2=nlp_tools.corpus([outfile2],nlp,prop=loadprop,ner=False,paired=True)
    return(c1,c2)
