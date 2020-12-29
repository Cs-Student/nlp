

import nltk
from nltk import RegexpTokenizer
from sklearn.metrics import confusion_matrix
filepath = 'Brown_train.txt'
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.util import ngrams
from sklearn.model_selection import KFold
import pandas as pd



def list_sen():
    op = open( filepath,'r',encoding="utf8", errors='ignore')
    sen_list = []
    for lin in op:
        line_split = lin.split(" ")
        temp_list = [("#","#"),("#","#")]
        for wrd in line_split:
            ar = wrd.split("_")
            if len(ar)<2:
                continue
            temp_list.append((ar[0],ar[1]))
        temp_list.append(("&","&"))
        sen_list.append(temp_list)
    return sen_list



def tag_dic(train_list):
    tag_dict = {}
    for sen in train_list:
        for (word,tag) in sen:
            word = word.lower()
            if tag not in tag_dict:
                tag_dict[tag] = {}
            if word not in tag_dict[tag]:
                tag_dict[tag][word] = 1
            tag_dict[tag][word] = tag_dict[tag][word]+1
    return tag_dict

def em_prob(tag_dict):
    emmision_prob = {}
    for key in tag_dict:
        if key not in emmision_prob:
            emmision_prob[key] = {}
        for word in tag_dict[key]:
            emmision_prob[key][word] = tag_dict[key][word]/sum(tag_dict[key].values())
    return emmision_prob


def big_prob(train_list):
    bigram_count = {}
    for sen in train_list:
        bigrams = list(nltk.bigrams(sen))
        for w1,w2 in bigrams:
            if w1[1] not in bigram_count:
                bigram_count[w1[1]] = {}
            if w2[1] not in bigram_count[w1[1]]:
                bigram_count[w1[1]][w2[1]] = 1
            bigram_count[w1[1]][w2[1]] +=1
    bigram_tag_prob = {}
    for key in bigram_count:
        if key not in bigram_tag_prob:
            bigram_tag_prob[key] = {}
        for tag_key in bigram_count[key]:
            temp_sum = sum(bigram_count[key].values())
            bigram_tag_prob[key][tag_key] = bigram_count[key][tag_key] / temp_sum
    return bigram_tag_prob


def tri_prob(train_list):
    trigram_count = {}
    for sen in train_list:
        trigrams = list(nltk.trigrams(sen))
        for w1,w2,w3 in trigrams:
            if (w1[1],w2[1]) not in trigram_count:
                trigram_count[(w1[1],w2[1])] = {}
            if w3[1] not in trigram_count[(w1[1],w2[1])]:
               trigram_count[(w1[1],w2[1])][w3[1]] = 1
            trigram_count[(w1[1],w2[1])][w3[1]] +=1
    trigram_tag_prob = {}
    for key in trigram_count:
        if key not in trigram_tag_prob:
            trigram_tag_prob[key] = {}
        for tag_key in trigram_count[key]:
            temp_sum = sum(trigram_count[key].values())
            trigram_tag_prob[key][tag_key] = trigram_count[key][tag_key] / temp_sum
    return trigram_tag_prob


def word_tags(train_list,test_list):
    word_tags_dict = {}
    for sen in train_list:
        for (word,tag) in sen:
            wrd = word.lower()
            if wrd not in word_tags_dict:
                word_tags_dict[wrd] = []
            if tag not in word_tags_dict[wrd]:
                word_tags_dict[wrd].append(tag)


    for sen in test_list:
        for (word,tag) in sen:
            wrd = word.lower()
            if wrd not in word_tags_dict:
                word_tags_dict[wrd] = []
            if tag not in word_tags_dict[wrd]:
                word_tags_dict[wrd].append(tag)
    return word_tags_dict

#%%

def split_test_word_tag(test_list):
    test_sent_words = []
    test_sent_tags = []
    for sen in test_list:
        temp_wrd = []
        temp_tag = []
        for wrd,tag in sen:
            temp_wrd.append(wrd.lower())
            temp_tag.append(tag)
        test_sent_words.append(temp_wrd)
        test_sent_tags.append(temp_tag)
    return test_sent_tags,test_sent_words


def predictions(test_sent_words,word_tags_dict,bigram_transi_prob,trigram_tag_prob,emmision_prob):
    predicted_tags = []
    for i in range(len(test_sent_words)):
        test_sen = test_sent_words[i]
        val = {}
        for j in range(len(test_sen)):
            wrd = test_sen[j]
            if j == 1 or j ==2:
                val[j] = {}
                tags = word_tags_dict[wrd]
                for tag in tags:
                    try:
                        val[j][tag] = ['#',bigram_transi_prob['#'][tag]*emmision_prob[tag][wrd]]
                    except:
                        val[j][tag] = ['#',0.00001]
            if j > 2:
                val[j] = {}
                curr_state_tags = word_tags_dict[wrd]
                prev_states_tags = list(val[j-1].keys())
                bef_prev_states_tags = list(val[j - 2].keys())
                for cur_tags in curr_state_tags:
                    tmp = []
                    for prev_tags in prev_states_tags:
                        for before_pr in bef_prev_states_tags:
                            try:
                                tmp.append(
                                    val[j-1][(before_pr,prev_tags)][1]*trigram_tag_prob[(prev_tags,cur_tags)][cur_tags]*emmision_prob[cur_tags][wrd]
                                )
                            except:
                                tmp.append(val[j-1][(prev_tags,cur_tags)][1]*0.0001)
                    max_idx = tmp.index(max(tmp))
                    max_prev_state_val = prev_states_tags[max_idx]
                    val[j][cur_tags] = [max_prev_state_val,max(tmp)]

        pred_tags = []
        total_steps_num = val.keys()
        last_step_num = max(total_steps_num)
        for bs in range(len(total_steps_num)):
            step_num = last_step_num - bs
            if step_num == last_step_num:
              pred_tags.append('&')
              pred_tags.append(val[step_num]['&'][0])
            if step_num<last_step_num and step_num>0:
              pred_tags.append(val[step_num][pred_tags[len(pred_tags)-1]][0])
        predicted_tags.append(list(reversed(pred_tags)))
    return predicted_tags

def pricsn(test_sent_tags,predicted_tags):
    right = 0
    wrong = 0
    for i in range(len(test_sent_tags)):
      tag_sen = test_sent_tags[i]
      pred = predicted_tags[i]
      for h in range(len(tag_sen)):
        if tag_sen[h] == pred[h]:
          right = right+1
        else:
          wrong = wrong +1
    prec = right/(right+wrong)
    rec = right/(right+wrong)
    f1 = 2*prec*rec/(prec+rec)
    return prec,rec,f1


def confusion_matrix(keys,pred,act):
    rows,col = (len(keys),len(keys))
    conf_arr = [[0 for i in range(col)] for j in range(rows)]
    for i in range(len(pred)):
        pred_sen = pred[i]
        act_sen = act[i]
        for j in range(len(pred_sen)):
            try:
                t1 = keys.index(pred_sen[j])
                t2 = keys.index(act_sen[j])
                conf_arr[t1][t2] += 1
            except:
                pass
    return conf_arr

def tag_prec(cm):
    true_pos = np.diag(cm)
    false_pos = np.sum(cm, axis=0) - true_pos
    false_neg = np.sum(cm, axis=1) - true_pos
    p=0
    p1=0
    for i in range(len(cm)):
        tp = true_pos[i]
        fp = false_pos[i]
        din = (tp+fp)
        if din !=0:
            p += tp/din
    for i in range(len(cm)):
        tp = true_pos[i]
        fn = false_neg[i]
        din = (tp+fn)
        if din !=0:
            p1 += tp/din
    prec = p/len(cm)
    rec = p1/len(cm)
    f1 = 2*prec*rec/(prec+rec)
    return prec,rec,f1


def main():
    mod_sen_list = list_sen()
    mod_sen_list = np.array(mod_sen_list)
    kf = KFold(n_splits=3)
    k = 1
    for train_index, test_index in kf.split(mod_sen_list):
        train_list, test_list = mod_sen_list[train_index], mod_sen_list[test_index]
        dic_tags = tag_dic(train_list)
        emsn_prob = em_prob(dic_tags)
        trigram_transi_prob = tri_prob(train_list)
        bigram_transi_prob = big_prob(train_list)
        word_tags_dict = word_tags(train_list,test_list)
        test_tags,test_words = split_test_word_tag(test_list)
        prd = predictions(test_words,word_tags_dict,bigram_transi_prob,trigram_transi_prob,emsn_prob)
        mat = confusion_matrix(list(dic_tags.keys()),prd,test_tags)
        # print(mat)
        sen_prec, sen_rec, sen_f1 = pricsn(test_tags, prd)
        prec,rec,f1 = tag_prec(mat)
        print('tag Precision on the test data is with bigrams at {} iter is {} '.format(k,prec))
        print('tag recall on the test data is with bigrams at {} iter is {} '.format(k,rec))
        print('tag f1_score on the test data is with bigrams at {} iter is {} '.format(k,f1))
        print('Sentence Precision on the test data is with bigrams at {} iter is {} '.format(k,sen_prec))
        print('Sentence Recall on the test data is with bigrams at {} iter is {} '.format(k,sen_rec))
        print('Sentence F1_score on the test data is with bigrams at {} iter is {} '.format(k,sen_f1))
        k+=1
main()



