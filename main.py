import os
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class TF_IDF():
    def __init__(self,
                 corpus_dir="corpus",
                 index=None,

                 ):
        self.corpus_dir = corpus_dir
        self.index = index

    def create_corpus(self):
        corpus_total = []
        for file in os.listdir(self.corpus_dir):
            corpus = []
            with open(os.path.join(self.corpus_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    line = re.sub("\s", "", line)  # 去掉空格(也去掉了换行符)
                    corpus += jieba.cut(line)
            corpus_total.append(" ".join(corpus))
        return corpus_total

    def create_tfidf(self):
        corpus_total = self.create_corpus()

        # 将文本中的词语转换为词频矩阵
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus_total)
        # 获取词袋中特征词词语
        word = vectorizer.get_feature_names()
        # 查看词频结果
        print(word)
        print(X.toarray())

        # 将词频矩阵X统计成TF-IDF值，# 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)
        print(tfidf.toarray())





if __name__ == "__main__":
    ifidf = TF_IDF()
    ifidf.create_tfidf()



