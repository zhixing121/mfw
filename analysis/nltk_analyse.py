"""用Python
进行机器学习及情感分析，需要用到两个主要的程序包：nltk
和
scikit - learn
nltk
主要负责处理特征提取（双词或多词搭配需要使用nltk
来做）和特征选择（需要nltk
提供的统计方法）。
scikit - learn
主要负责分类算法，评价分类效果，进行分类等任务。

接下来会有四篇文章按照以下步骤来实现机器学习的情感分析。
1.
特征提取和特征选择（选择最佳特征）
2.
赋予类标签，分割开发集和测试集
3.
构建分类器，检验分类准确度，选择最佳分类算法
4.
存储和使用最佳分类器进行分类，分类结果为概率值

首先是特征提取和选择
一、特征提取方法
1."""

import pickle
import itertools
import sklearn
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
# 把所有词作为特征


def bag_of_words(words):
    return dict([(word, True) for word in words])


"""返回的是字典类型，这是nltk
处理情感分类的一个标准形式。

2."""
# 把双词搭配（bigrams）作为特征



def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n)  # 使用了卡方统计的方法，选择排名前1000的双词
    return bag_of_words(bigrams)


"""除了可以使用卡方统计来选择信息量丰富的双词搭配，还可以使用其它的方法，比如互信息（PMI）。而排名前1000也只是人工选择的阈值，可以随意选择其它值，可经过测试一步步找到最优值。

3."""


# 把所有词和双词搭配一起作为特征


def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)  # 所有词和（信息量大的）双词搭配一起作为特征


"""二、特征选择方法
有了提取特征的方法后，我们就可以提取特征来进行分类学习了。但一般来说，太多的特征会降低分类的准确度，所以需要使用一定的方法，来“选择”出信息量最丰富的特征，再使用这些特征来分类。
特征选择遵循如下步骤：
1.
计算出整个语料里面每个词的信息量
2.
根据信息量进行倒序排序，选择排名靠前的信息量的词
3.
把这些词作为特征

1.
计算出整个语料里面每个词的信息量
1.1"""
# 计算整个语料里面每个词的信息量



def create_word_scores():
    posWords = pickle.load(open('D:/code/sentiment_test/pos_review.pkl', 'r'))
    negWords = pickle.load(open('D:/code/sentiment_test/neg_review.pkl', 'r'))

    posWords = list(itertools.chain(*posWords))  # 把多维数组解链成一维数组
    negWords = list(itertools.chain(*negWords))  # 同理

    word_fd = FreqDist()  # 可统计所有词的词频
    cond_word_fd = ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频
    for word in posWords:
        word_fd.inc(word)
        cond_word_fd['pos'].inc(word)
    for word in negWords:
        word_fd.inc(word)
        cond_word_fd['neg'].inc(word)

    pos_word_count = cond_word_fd['pos'].N()  # 积极词的数量
    neg_word_count = cond_word_fd['neg'].N()  # 消极词的数量
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count),
                                               total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count),
                                               total_word_count)  # 同理
        word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量

    return word_scores  # 包括了每个词和这个词的信息量


# 1.2计算整个语料里面每个词和双词搭配的信息量


def create_word_bigram_scores():
    posdata = pickle.load(open('D:/code/sentiment_test/pos_review.pkl', 'r'))
    negdata = pickle.load(open('D:/code/sentiment_test/neg_review.pkl', 'r'))

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams  # 词和双词搭配
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd.inc(word)
        cond_word_fd['pos'].inc(word)
    for word in neg:
        word_fd.inc(word)
        cond_word_fd['neg'].inc(word)

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


# 2.根据信息量进行倒序排序，选择排名靠前的信息量的词


def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[
                :number]  # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_words = set([w for w, s in best_vals])
    return best_words


# 然后需要对find_best_words赋值，如下：
word_scores_1 = create_word_scores()
word_scores_2 = create_word_bigram_scores()


# 3.把选出的这些词作为特征（这就是选择了信息量丰富的特征）
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


"""第一步，载入数据。
要做情感分析，首要的是要有数据。
数据是人工已经标注好的文本，有一部分积极的文本，一部分是消极的文本。
文本是已经分词去停用词的商品评论，形式大致如下：[[word11, word12, ... word1n], [word21, word22, ... , word2n], ... , [wordn1, wordn2, ... , wordnn]]
这是一个多维数组，每一维是一条评论，每条评论是已经又该评论的分词组成。
#! /usr/bin/env python2.7
#coding=utf-8
"""
pos_review = pickle.load(open('D:/code/sentiment_test/pos_review.pkl','r'))
neg_review = pickle.load(open('D:/code/sentiment_test/neg_review.pkl','r'))
# 我用pickle 存储了相应的数据，这里直接载入即可。
# 第二步，使积极文本的数量和消极文本的数量一样。
from random import shuffle


shuffle(pos_review) #把积极文本的排列随机化
size = int(len(pos_review)/2 - 18)
pos = pos_review[:size]
neg = neg_review
# 我这里积极文本的数据恰好是消极文本的2倍还多18个，所以为了平衡两者数量才这样做。

# 第三步，赋予类标签。
def pos_features(feature_extraction_method):
    posFeatures = []
    for i in pos:
        posWords = [feature_extraction_method(i),'pos'] #为积极文本赋予"pos"
        posFeatures.append(posWords)
    return posFeatures

def neg_features(feature_extraction_method):
    negFeatures = []
    for j in neg:
        negWords = [feature_extraction_method(j),'neg'] #为消极文本赋予"neg"
        negFeatures.append(negWords)
    return negFeatures
# 这个需要用特征选择方法把文本特征化之后再赋予类标签。

# 第四步、把特征化之后的数据数据分割为开发集和测试集
train = posFeatures[174:]+negFeatures[174:]
devtest = posFeatures[124:174]+negFeatures[124:174]
test = posFeatures[:124]+negFeatures[:124]
# 这里把前124个数据作为测试集，中间50个数据作为开发测试集，最后剩下的大部分数据作为训练集。




"""三、检测哪中特征选择方法更优
见构建分类器，检验分类准确度，选择最佳分类算法
在把文本转化为特征表示，并且分割为开发集和测试集之后，我们就需要针对开发集进行情感分类器的开发。测试集就放在一边暂时不管。
开发集分为训练集（Training Set）和开发测试集（Dev-Test Set）。训练集用于训练分类器，而开发测试集用于检验分类器的准确度。
为了检验分类器准确度，必须对比“分类器的分类结果”和“人工标注的正确结果”之间的差异。
所以第一步，是要把开发测试集中，人工标注的标签和数据分割开来。第二步是使用训练集训练分类器；第三步是用分类器对开发测试集里面的数据进行分类，给出分类预测的标签；第四步是对比分类标签和人工标注的差异，计算出准确度。
"""
# 一、分割人工标注的标签和数据
dev, tag_dev = zip(*devtest) #把开发测试集（已经经过特征化和赋予标签了）分为数据和标签


# 二到四、可以用一个函数来做
def score(classifier):
    classifier = SklearnClassifier(classifier) #在nltk 中使用scikit-learn 的接口
    classifier.train(train) #训练分类器

    pred = classifier.batch_classify(testSet) #对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score(tag_test, pred) #对比分类预测结果和人工标注的正确结果，给出分类器准确度

# 之后我们就可以简单的检验不同分类器和不同的特征选择的结果
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


posFeatures = pos_features(bag_of_words) #使用所有词作为特征
negFeatures = neg_features(bag_of_words)


print('BernoulliNB`s accuracy is %f' %score(BernoulliNB()))
print('MultinomiaNB`s accuracy is %f' %score(MultinomialNB()))
print('LogisticRegression`s accuracy is %f' %score(LogisticRegression()))
print('SVC`s accuracy is %f' %score(SVC()))
print('LinearSVC`s accuracy is %f' %score(LinearSVC()))
print('NuSVC`s accuracy is %f' %score(NuSVC()))
# 1. 我选择了六个分类算法，可以先看到它们在使用所有词作特征时的效果：
"""BernoulliNB`s accuracy is 0.790000
MultinomiaNB`s accuracy is 0.810000
LogisticRegression`s accuracy is 0.710000
SVC`s accuracy is 0.650000
LinearSVC`s accuracy is 0.680000
NuSVC`s accuracy is 0.740000
"""
# 2. 再看使用双词搭配作特征时的效果（代码改动如下地方即可）
posFeatures = pos_features(bigrams)
negFeatures = neg_features(bigrams)
"""结果如下：
BernoulliNB`s accuracy is 0.710000
MultinomiaNB`s accuracy is 0.750000
LogisticRegression`s accuracy is 0.790000
SVC`s accuracy is 0.750000
LinearSVC`s accuracy is 0.770000
NuSVC`s accuracy is 0.780000
"""
# 3. 再看使用所有词加上双词搭配作特征的效果
posFeatures = pos_features(bigram_words)
negFeatures = neg_features(bigram_words)
"""结果如下：
BernoulliNB`s accuracy is 0.780000
MultinomiaNB`s accuracy is 0.780000
LogisticRegression`s accuracy is 0.780000
SVC`s accuracy is 0.600000
LinearSVC`s accuracy is 0.790000
NuSVC`s accuracy is 0.790000
"""
"""可以看到在不选择信息量丰富的特征时，仅仅使用全部的词或双词搭配作为特征，分类器的效果并不理想。
接下来将使用卡方统计量（Chi-square）来选择信息量丰富的特征，再用这些特征来训练分类器。
4. 计算信息量丰富的词，并以此作为分类特征
"""
word_scores = create_word_scores()
best_words = find_best_words(word_scores, 1500) #选择信息量最丰富的1500个的特征


posFeatures = pos_features(best_word_features)
negFeatures = neg_features(best_word_features)
"""结果如下：
BernoulliNB`s accuracy is 0.870000
MultinomiaNB`s accuracy is 0.860000
LogisticRegression`s accuracy is 0.730000
SVC`s accuracy is 0.770000
LinearSVC`s accuracy is 0.720000
NuSVC`s accuracy is 0.780000
可见贝叶斯分类器的分类效果有了很大提升。
"""
# 5. 计算信息量丰富的词和双词搭配，并以此作为特征
word_scores = create_word_bigram_scores()
best_words = find_best_words(word_scores, 1500) #选择信息量最丰富的1500个的特征


posFeatures = pos_features(best_word_features)
negFeatures = neg_features(best_word_features)
"""结果如下：
BernoulliNB`s accuracy is 0.910000
MultinomiaNB`s accuracy is 0.860000
LogisticRegression`s accuracy is 0.800000
SVC`s accuracy is 0.800000
LinearSVC`s accuracy is 0.750000
NuSVC`s accuracy is 0.860000
可以发现贝努利的贝叶斯分类器效果继续提升，同时NuSVC 也有很大的提升。
"""
# 此时，我们选用BernoulliNB、MultinomiaNB、NuSVC 作为候选分类器，使用词和双词搭配作为特征提取方式，测试不同的特征维度的效果。
dimension = ['500','1000','1500','2000','2500','3000']

for d in dimension:
    word_scores = create_word_scores_bigram()
    best_words = find_best_words(word_scores, int(d))

    posFeatures = pos_features(best_word_features)
    negFeatures = neg_features(best_word_features)


    train = posFeatures[174:]+negFeatures[174:]
    devtest = posFeatures[124:174]+negFeatures[124:174]
    test = posFeatures[:124]+negFeatures[:124]
    dev, tag_dev = zip(*devtest)

    print('Feature number %f' %d)
    print('BernoulliNB`s accuracy is %f' %score(BernoulliNB()))
    print('MultinomiaNB`s accuracy is %f' %score(MultinomialNB()))
    print('LogisticRegression`s accuracy is %f' %score(LogisticRegression()))
    print('SVC`s accuracy is %f' %score(SVC()))
    print('LinearSVC`s accuracy is %f' %score(LinearSVC()))
    print('NuSVC`s accuracy is %f' %score(NuSVC()))
    # print
"""结果如下（很长。。）：
Feature number 500
BernoulliNB`s accuracy is 0.880000
MultinomiaNB`s accuracy is 0.850000
LogisticRegression`s accuracy is 0.740000
SVC`s accuracy is 0.840000
LinearSVC`s accuracy is 0.700000
NuSVC`s accuracy is 0.810000

Feature number 1000
BernoulliNB`s accuracy is 0.860000
MultinomiaNB`s accuracy is 0.850000
LogisticRegression`s accuracy is 0.750000
SVC`s accuracy is 0.800000
LinearSVC`s accuracy is 0.720000
NuSVC`s accuracy is 0.760000

Feature number 1500
BernoulliNB`s accuracy is 0.870000
MultinomiaNB`s accuracy is 0.860000
LogisticRegression`s accuracy is 0.770000
SVC`s accuracy is 0.770000
LinearSVC`s accuracy is 0.750000
NuSVC`s accuracy is 0.790000

Feature number 2000
BernoulliNB`s accuracy is 0.870000
MultinomiaNB`s accuracy is 0.850000
LogisticRegression`s accuracy is 0.770000
SVC`s accuracy is 0.690000
LinearSVC`s accuracy is 0.700000
NuSVC`s accuracy is 0.800000

Feature number 2500
BernoulliNB`s accuracy is 0.850000
MultinomiaNB`s accuracy is 0.830000
LogisticRegression`s accuracy is 0.780000
SVC`s accuracy is 0.700000
LinearSVC`s accuracy is 0.730000
NuSVC`s accuracy is 0.800000

Feature number 3000
BernoulliNB`s accuracy is 0.850000
MultinomiaNB`s accuracy is 0.830000
LogisticRegression`s accuracy is 0.780000
SVC`s accuracy is 0.690000
LinearSVC`s accuracy is 0.710000
NuSVC`s accuracy is 0.800000

把上面的所有测试结果进行综合可汇总如下：
不同分类器的不同特征选择方法效果
bag_of_words	bigrams	bigram_words	best_word_feature	best_word_bigram_feature
BernoulliNB	0.79	0.71	0.78	0.87	0.91
MultinomiaNB	0.81	0.75	0.78	0.86	0.86
LogisticRegression	0.71	0.79	0.78	0.73	0.8
SVC	0.65	0.75	0.6	0.77	0.8
LinearSVC	0.68	0.77	0.79	0.72	0.75
NuSVC	0.74	0.78	0.79	0.78	0.86

候选分类器在不同特征维度下的效果
500	1000	1500	2000	2500	3000
BernoulliNB	0.88	0.86	0.87	0.87	0.85	0.85
MultinomiaNB	0.85	0.85	0.86	0.85	0.83	0.83
NuSVC	0.81	0.76	0.79	0.7	0.8	0.8



综合来看，可以看出特征维数在500 或 1500的时候，分类器的效果是最优的。

所以在经过上面一系列的分析之后，可以得出如下的结论：
Bernoulli 朴素贝叶斯分类器效果最佳
词和双词搭配作为特征时效果最好
当特征维数为1500时效果最好
"""

# 为了不用每次分类之前都要训练一次数据，所以可以在用开发集找出最佳分类器后，把最佳分类器存储下来以便以后使用。然后再使用这个分类器对文本进行分类。

# 一、使用测试集测试分类器的最终效果
word_scores = create_word_bigram_scores() #使用词和双词搭配作为特征
best_words = find_best_words(word_scores, 1500) #特征维度1500

posFeatures = pos_features(best_word_features)
negFeatures = neg_features(best_word_features)

trainSet = posFeatures[:500] + negFeatures[:500] #使用了更多数据
testSet = posFeatures[500:] + negFeatures[500:]
test, tag_test = zip(*testSet)

def final_score(classifier):
    classifier = SklearnClassifier(classifier)
    classifier.train(trainSet)
    pred = classifier.batch_classify(test)
    return accuracy_score(tag_test, pred)

print(final_score(BernoulliNB()) )#使用开发集中得出的最佳分类器
"""其结果是很给力的：
0.979166666667
"""
# 二、把分类器存储下来
# （存储分类器和前面没有区别，只是使用了更多的训练数据以便分类器更为准确）
word_scores = create_word_bigram_scores()
best_words = find_best_words(word_scores, 1500)

posFeatures = pos_features(best_word_features)
negFeatures = neg_features(best_word_features)

trainSet = posFeatures + negFeatures

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(trainSet)
pickle.dump(BernoulliNB_classifier, open('D:/code/sentiment_test/classifier.pkl','w'))

# 在存储了分类器之后，就可以使用该分类器来进行分类了。
# 三、使用分类器进行分类，并给出概率值
# 给出概率值的意思是用分类器判断一条评论文本的积极概率和消极概率。给出类别也是可以的，也就是可以直接用分类器判断一条评论文本是积极的还是消极的，但概率可以提供更多的参考信息，对以后判断评论的效用也是比单纯给出类别更有帮助。

# 1. 把文本变为特征表示的形式
# 要对文本进行分类，首先要把文本变成特征表示的形式。而且要选择和分类器一样的特征提取方法。
#! /usr/bin/env python2.7
#coding=utf-8


moto = pickle.load(open('D:/code/review_set/senti_review_pkl/moto_senti_seg.pkl','r')) #载入文本数据


def extract_features(data):
    feat = []
    for i in data:
        feat.append(best_word_features(i))
    return feat


moto_features = extract_features(moto) #把文本转化为特征表示的形式
# 注：载入的文本数据已经经过分词和去停用词处理。

# 2. 对文本进行分类，给出概率值



clf = pickle.load(open('D:/code/sentiment_test/classifier.pkl')) #载入分类器

pred = clf.batch_prob_classify(moto_features) #该方法是计算分类概率值的
p_file = open('D:/code/sentiment_test/score/Motorala/moto_ml_socre.txt','w') #把结果写入文档
for i in pred:
    p_file.write(str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n')
p_file.close()
"""最后分类结果如下图：
Python 文本挖掘：使用机器学习方法进行情感分析（四、使用分类器进行分类） - rzcoding - Explore in Data
 前面是积极概率，后面是消极概率
折腾了这么久就为了搞这么一个文件出来。。。这伤不起的节奏已经无人阻挡了吗。。。
不过这个结果确实比词典匹配准确很多，也算欣慰了。。。
"""