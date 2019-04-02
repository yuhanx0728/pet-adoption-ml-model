import pandas, random, string, re

dataset = pandas.read_csv("text-train.csv")
text_list = dataset["Description"].tolist()
text_processed = [''.join(t if not t.isdigit() else " <num> " for t in str(text).translate(str.maketrans("","", string.punctuation)).strip() ).strip().lower() for text in text_list]
text_processed = [''.join(entry+" " for entry in re.split(r'([\u4e00-\u9fff])',text)).strip() for text in text_processed]
dataset["Description"] = text_processed
dataset.to_csv("processed.csv")
