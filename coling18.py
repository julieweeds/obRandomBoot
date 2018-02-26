__author__="juliewe"

#code to generate characteristic sets including bootstrapping

import pandas as pd
import matplotlib as plt
from collections import defaultdict
import random
import nlp_tools
import spacy,operator
import logging,os
from generateCorpus import *
from CharacterisingFunctions import *
from time import time
from multiprocessing import Pool



            

def cdump(cset,filename):
        
    path=os.path.join("csets",filename)
        
    with open(path,"w") as outstream:
        logging.info("Dumping characteristic set at {}".format(path))
        for(word,score) in cset:
            outstream.write(word+"\t"+str(score)+"\n")
            
            
def do_surprises(args):
    dist=args[0]
    hf_theft=args[1]
    info=args[2]
    res=improved_compute_surprises(dist,hf_theft[info['ftype']+":"+info['smoothing']],info['m'],display=False)
    filename=info['ftype']+":"+info['m']+str(info['smoothing'])+"_"+str(info['key'])+"__"+str(info['thisseed'])  
    cdump(res,filename)
    
    return res
    
def do_bootstrap(args):
    corpusAlabels=args[0]
    corpusAworddocdict=args[1]
    csize=args[2]
    cdist=args[3]
    info=args[4]
    starttime=time()
    logging.info("{}: Bootstrapping for {}, {}: balanced = {}".format(info['count'],info['m'],info['key'],info['b']))
    res=bootstrap_compare((corpusAlabels,corpusAworddocdict),(csize,cdist),ftype=info['m'],repeats=info['repeats'],balanced=info['b'],size=info['size'])    
    #print(characteristic_sets[name][key])
    filename=info['m']+":"+str(info['b'])+"_"+str(info['key'])+"__"+str(info['thisseed'])
    cdump(res,filename)

    timetaken=time()-starttime
    logging.info("Time taken for combination {} is {}".format(info['count'],timetaken))
    return res

if __name__=="__main__":
    nlp=spacy.load('en')
    logging.basicConfig(level=logging.INFO)
    parentdir=""
    allreqlist=[('deft_offcat','theft'),('year:min',1800),('year:max',1820),('obv_role',['def','wv'])]
    worddatafile=os.path.join(parentdir,"obv_words_v2_28-01-2017.tsv")
    trialdatafile=os.path.join(parentdir,"obv_defendants_trials.tsv")

    worddata=pd.read_csv(worddatafile,sep='\t')
    trialdata=pd.read_csv(trialdatafile,sep='\t')
     
    threads=24
    runs=5                          
    seeds=[19,23,29,37,13]
    splits=[5,10,15,20,25,30,35,40,45,50]
    smoothing=[0,0.5]
    info={}

    info['k']=10000 #top k words (by frequency) to consider
    logging.info("Initialisation complete")
    testing=False
    

    for run in range(0,runs):
        info['thisseed']=seeds[run]


        if testing:
            splits=splits[:1]

        #generate randomly split corpora                       
        corpora={}
        for prop in splits:
            (c1,c2)=random_split(worddata,trialdata,allreqlist,prop=prop,cache=True,testing=testing,seed=info['thisseed'])
            corpora[prop]=c1
            corpora[100-prop]=c2
                               
                               
        #generate term and doc frequency distributions for whole corpus for comparison  
        hf_theft={}
        for smooth in smoothing:
            hf_theft["termfreq:"+str(smooth)]=find_hfw_dist(list(corpora.values())[:2], k=info['k'],smoothing=smooth)
            hf_theft["docfreq:"+str(smooth)]=find_hfw_dist(list(corpora.values())[:2],k=info['k'],ftype='docfreq',smoothing=smooth)
        size=0
        for c in list(corpora.values())[:2]:
            size+=len(set(c.labels))
        info['size']=size
        
        measures=['llr','pmi','kl','jsd']
        ftypes=["termfreq","docfreq"]
       
        #generate characteristic sets for non-bootstrapped measures  
        
        inputs=[]
        filenames=[]
        for ftype in ftypes:
            info['ftype']=ftype
            random_dists={}
            
            for key,corpus in corpora.items():
                random_dists[key]=find_hfw_dist([corpus],k=info['k'],ftype=ftype)
            for smooth in smoothing:
                info['smoothing']=str(smooth)
                for measure in measures:
                    info['m']=measure
                    for key,dist in random_dists.items():
                        info['key']=key
                        #do_surprises(dist,hf_theft,info)
                        filenames.append(info['m']+str(info['smoothing'])+"_"+str(info['key'])+"__"+str(info['thisseed']))
                        inputs.append(((dist[0],list(dist[1])),hf_theft,dict(info)))
        mappool=Pool(processes=threads)
        results=mappool.map(do_surprises,inputs)
        mappool.close()  
        for filename,res in zip(filenames,results):
            logging.info("{}:{}".format(filename,len(res)))
        
        #bootstrapped measures
        bsmeasures=["termfreq","docfreq"]
        bsmeasures=[]
        balanced=[False,True]

        if testing:
            info['repeats']=10
        else:
            info['repeats']=2000                          
                                      

        info['count']=0
        info['smoothing']='0'
        
        inputs=[]
        filenames=[]
       
        for key, corpusA in corpora.items():
            info['key']=key
    
            for m in bsmeasures:
                info['m']=m
                for b in balanced:
                    info['b']=b
                    
                    info['count']+=1
                    #do_bootstrap(corpusA,hf_theft,info)
                    compsize,compdist=hf_theft[info['m']+":"+info['smoothing']]
                    copy=corpusA.copy()
                    corpusAlabels=list(copy[0])
                    corpusAworddocdict=dict(copy[1])
                    inputs.append((corpusAlabels,corpusAworddocdict,compsize,list(compdist),dict(info)))
                    filenames.append(info['m']+":"+str(info['b'])+"_"+str(info['key'])+"__"+str(info['thisseed']))
        mappool=Pool(processes=threads)
        results=mappool.map(do_bootstrap,inputs)
        mappool.close()
        
        for res,filename in zip(results,filenames):
            #cdump(res,filename)
            logging.info("{}:{}".format(filename,len(res)))