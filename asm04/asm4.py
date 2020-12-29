
import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

top = 5

def get_L1():
    f = open("english-hindi-dictionary.txt", "r")
    eng_to_hin = {}
    for x in f:
        words = x.strip().split("|||")
        eng = lemmatizer.lemmatize(words[0].strip())
        hin = words[1][:words[1].find("\n")].strip()
        if eng not in eng_to_hin:
            eng_to_hin[eng] = list()
        eng_to_hin[eng].append(hin)

    df = pd.read_csv("BingLiu.csv")
    ar = df.values
    bing_eng_hin_l1 = {}
    for row in ar:
        row = row[0].split("\t")
        wrd = row[0].strip()
        pol = row[1].strip()
        wrd_bing = lemmatizer.lemmatize(wrd)
        key = (wrd_bing,pol)
        if wrd_bing in eng_to_hin:
            if wrd_bing not in bing_eng_hin_l1:
                bing_eng_hin_l1[key] = list()
            for hin in eng_to_hin[wrd_bing]:
                if not str.isalpha(hin):
                    bing_eng_hin_l1[key].append(hin)
    return bing_eng_hin_l1,eng_to_hin

def get_w2v_models():
    data_file = "english.txt"
    def read_input_eng(input_file):
        with open (input_file, 'r') as f:
            for i, line in enumerate (f):
                yield gensim.utils.simple_preprocess(line)
    eng_documents = list (read_input_eng (data_file))
    data_file = "hindi.txt"
    def read_input_hin(input_file):
        with open (input_file, 'r') as f:
            for i, line in enumerate (f):
                if "\n" in line:
                    line = line[:line.find("\n")].strip()
                yield  line.split(" ")
    hin_documents = list (read_input_hin (data_file))
    model_w2v_en = gensim.models.Word2Vec(eng_documents)
    model_w2v_en.train(eng_documents,total_examples=len(eng_documents),epochs=150)
    model_w2v_hi = gensim.models.Word2Vec (hin_documents)
    model_w2v_hi.train(hin_documents,total_examples=len(hin_documents),epochs=150)
    return model_w2v_en,model_w2v_hi


def get_glove_models():
    word2vec_output_file = 'word2vec_eng.txt'
    # glove2word2vec("vectors.txt", word2vec_output_file)
    model_glove_en = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    word2vec_output_file = 'word2vec.txt'
    # glove2word2vec(glove_input_file, word2vec_output_file)
    model_glove_hi = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    return model_glove_en,model_glove_hi

def iterate(bing_eng_hin_l1,model_w2v_hi,model_glove_en,model_glove_hi,eng_to_hin,model_w2v_en):
    top = 5
    ans = {}
    for key in bing_eng_hin_l1:
            engList = []
            hinList = []
            eng_wrd = key[0]

            if eng_wrd in model_w2v_en.wv.vocab:
                engList = model_w2v_en.most_similar(eng_wrd,topn=top)
            if eng_wrd in model_glove_en.wv.vocab:
                engList += model_glove_en.most_similar(eng_wrd,topn=top)
            for i in range(len(engList)):
                engList[i] = (engList[i], key[1])

            hin_temp_list = bing_eng_hin_l1[key]

            for hin_wrd in hin_temp_list:
                if hin_wrd in model_w2v_hi.wv.vocab:
                    hinList += model_w2v_hi.most_similar(hin_wrd,topn=top)
                if hin_wrd in model_glove_hi.wv.vocab:
                    hinList += model_glove_hi.most_similar(hin_wrd,topn=top)

            for eng in engList:
                engtemp = lemmatizer.lemmatize(eng[0][0])
                for hin in hinList:
                    if engtemp in eng_to_hin:
                        if hin[0] in eng_to_hin[engtemp]:
                            if (engtemp,eng[1]) not in bing_eng_hin_l1:
                                # bing_eng_hin_l1[(engtemp,eng[1])] = hin[0]
                                ans[(engtemp,eng[1])] = hin[0]
    return ans

def add_keys(new,old):
    for key in new:
        if key not in old:
            old[key] = [new[key]]
    return old

def main():
    bing_eng_hin_l1,eng_to_hin = get_L1()
    print("Initial length of L1 : {}".format(len(bing_eng_hin_l1)))
    model_w2v_en,model_w2v_hi = get_w2v_models()
    model_glove_en,model_glove_hi = get_glove_models()
    # print(model_glove_en.most_similar("Good",topn=5))
    print("new words generated in L1")
    ans = iterate(bing_eng_hin_l1,model_w2v_hi,model_glove_en,model_glove_hi,eng_to_hin,model_w2v_en)
    length=0
    while(len(ans)>0):
        length += len(ans)
        bing_eng_hin_l1 = add_keys(ans, bing_eng_hin_l1)
        ans = iterate(bing_eng_hin_l1, model_w2v_hi, model_glove_en, model_glove_hi, eng_to_hin, model_w2v_en)
        for key in ans:
            print("{} : {}".format(key,ans[key]))
    print(length)
main()


