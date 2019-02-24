sentiment_score_text = r"F:\mfw\BosonNLP_sentiment_score\BosonNLP_sentiment_score.txt"
with open(sentiment_score_text,'r',encoding='utf-8') as f:
    for line in f.readlines():
        print(line)
f.close()