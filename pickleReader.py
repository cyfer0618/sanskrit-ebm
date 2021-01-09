import pickle
import os

#In case of any clarification, email to - amrith@iitkgp.ac.in
# The latest modifications can be found at: www.amrithkrishna.com 
# Class definition for pickle files
# currently the pickle files can be opened only with Python 3



lemm2idx = {}
idx2lemm = []
lemm2count = {}
#lemm_lemm = [] 

def add_lemm(lemm):
    if lemm not in lemm2idx:
        idx2lemm.append(lemm)
        lemm2idx[lemm] = len(idx2lemm) - 1
        lemm2count[lemm] = 0
    return lemm2idx[lemm]

def count_lemm(lemm):
    lemm2count[lemm] += 1

cngg2idx = {}
idx2cngg = []
cngg2count = {}

def add_cngg(cngg):
    if cngg not in cngg2idx:
        idx2cngg.append(cngg)
        cngg2idx[cngg] = len(idx2cngg) - 1
        cngg2count[cngg] = 0
    return cngg2idx[cngg]

def count_cngg(cngg):
    cngg2count[cngg] += 1


#lemma_cng
lem_cng2idx = {}
idx2lem_cng = []
lem_cng2count = {}

def add_lem_cng(lem_cng):
    if lem_cng not in lem_cng2idx:
        idx2lem_cng.append(lem_cng)
        lem_cng2idx[lem_cng] = len(idx2lem_cng) - 1
        lem_cng2count[lem_cng] = 0
    return lem_cng2idx[lem_cng]

def count_lem_cng(lem_cng):
    lem_cng2count[lem_cng] += 1

thisdir = os.getcwd()
print(thisdir)
ndir = thisdir +'/DCS_pick'
print(ndir)
arr_txt = [x for x in os.listdir(ndir) if x.endswith(".p")]
#print(arr_txt)
i = 0
for a in range(len(arr_txt)):
    fil = ndir+"/"+arr_txt[a]
    #print(fil)
    output_load = pickle.load(open(fil, "rb"), encoding='utf-8')
    print(output_load)
    print('Sentence Id:',output_load.sent_id)
    print('Sentence:',output_load.sentence)
    print('Chunks:',output_load.dcs_chunks)
    print('Lemmas:',output_load.lemmas)
    print('Morphological class (CNG):',output_load.cng)
    for x in range(len(output_load.lemmas)):
        lemm = output_load.lemmas
        cngg = output_load.cng
        for y in range(len(lemm[x])):
            add_lemm(lemm[x][y])
            add_cngg(cngg[x][y])
            count_lemm(lemm[x][y])
            count_cngg(cngg[x][y])
            lem_cng = lemm[x][y]+"_"+cngg[x][y]
            add_lem_cng(lem_cng)
            count_lem_cng(lem_cng)


    # if i == 1:
    #     break
    # i += 1


#print(lemm2idx)
#print(lemm2count)
#print(cngg2idx)
#print(cngg2count)
#exit(1)

i = 0
lem_size = len(idx2lemm)
cng_size = len(idx2cngg)
lem_cng_size = len(idx2lem_cng)
lemm_lemm = [[0 for i in range(lem_size)] for j in range(lem_size)]
cngg_cngg = [[0 for i in range(cng_size)] for j in range(cng_size)]
lemm_cngg = [[0 for i in range(cng_size)] for j in range(lem_size)]
cngg_lemm = [[0 for i in range(lem_size)] for j in range(cng_size)]
lem_cng_lemm = [[0 for i in range(lem_size)] for j in range(lem_cng_size)]
lem_cng_cngg = [[0 for i in range(cng_size)] for j in range(lem_cng_size)]
lem_cng_lem_cng = [[0 for i in range(lem_cng_size)] for j in range(lem_cng_size)]
for a in range(len(arr_txt)):
    fil = ndir+"/"+arr_txt[a]
    #print(fil)
    output_load = pickle.load(open(fil, "rb"), encoding='utf-8')
    print(output_load)
    lemm = output_load.lemmas
    cngg = output_load.cng
    lemm_list = []
    cngg_list = []
    for x in range(len(output_load.lemmas)):
        for y in range(len(lemm[x])):
            lemm_list.append(lemm[x][y])
            cngg_list.append(cngg[x][y])
    #lemma_lemma and CNG_CNG
    for x in range(len(lemm_list)):
        for y in range(x+1,len(lemm_list)):
            lemm_lemm[lemm2idx[lemm_list[x]]][lemm2idx[lemm_list[y]]] += 1
            if lemm_list[x] != lemm_list[y]:
                lemm_lemm[lemm2idx[lemm_list[y]]][lemm2idx[lemm_list[x]]] += 1

            cngg_cngg[cngg2idx[cngg_list[x]]][cngg2idx[cngg_list[y]]] += 1
            if cngg_list[x] != cngg_list[y]:
                cngg_cngg[cngg2idx[cngg_list[y]]][cngg2idx[cngg_list[x]]] += 1

            lem_cng_x =  lemm_list[x] +"_"+ cngg_list[x]
            lem_cng_y =  lemm_list[y] +"_"+ cngg_list[y]

            lem_cng_lem_cng[lem_cng2idx[lem_cng_x]][lem_cng2idx[lem_cng_y]] += 1
            if lem_cng_x != lem_cng_y:
                lem_cng_lem_cng[lem_cng2idx[lem_cng_y]][lem_cng2idx[lem_cng_x]] += 1
            #print(x,y,lemm_lemm,cngg_cngg)

    #lemma_CNG and CNG_lemma
    for x in range(len(lemm_list)):
        for y in range(len(lemm_list)):
            if x == y:
                continue
            lemm_cngg[lemm2idx[lemm_list[x]]][cngg2idx[cngg_list[y]]] += 1
            # if lemm_list[x] != lemm_list[y]:
            #     lemm_lemm[lemm2idx[lemm_list[y]]][lemm2idx[lemm_list[x]]] += 1

            cngg_lemm[cngg2idx[cngg_list[x]]][lemm2idx[lemm_list[y]]] += 1
            # if cngg_list[x] != cngg_list[y]:
            #      cngg_cngg[cngg2idx[cngg_list[y]]][cngg2idx[cngg_list[x]]] += 1
            #print("Lemma to CNG Matrix ",lemm_cngg)
            #print("CNG to Lemma Matrix ",cngg_lemm)

    #lemma_CNG and CNG_lemma
    for x in range(len(lemm_list)):
        for y in range(len(lemm_list)):
            if x == y:
                continue
            lem_cng = lemm_list[x]+"_"+cngg_list[x]
            lem_cng_lemm[lem_cng2idx[lem_cng]][lemm2idx[lemm_list[y]]] += 1
            # if lemm_list[x] != lemm_list[y]:
            #     lemm_lemm[lemm2idx[lemm_list[y]]][lemm2idx[lemm_list[x]]] += 1

            lem_cng_cngg[lem_cng2idx[lem_cng]][cngg2idx[cngg_list[y]]] += 1


    # if i == 1:
    #     break
    # i += 1





print("Lemma to Index ",lemm2idx)
print("Lemma to Count ",lemm2count)
print("Lemma_CNG to Index ",lem_cng2idx)
print("Lemma_CNG to Count ",lem_cng2count)
print("Lemma to Lemma Matrix ",lemm_lemm)
print("CNG to CNG Matrix ",cngg_cngg)
print("Lemma to CNG Matrix ",lemm_cngg)
print("CNG to Lemma Matrix ",cngg_lemm)
print("Lemma_CNG to CNG Matrix ",lem_cng_cngg)
print("Lemma_CNG to Lemma Matrix ",lem_cng_lemm)
print("Lemma_CNG to Lemma_CNG Matrix ",lem_cng_lem_cng)

