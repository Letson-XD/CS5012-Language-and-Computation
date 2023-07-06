import nltk
from nltk import FeatureChartParser

fcfg = nltk.data.load('/grammar.fcfg')
parser = FeatureChartParser(fcfg)

def parse_text(text):
	examples = text.splitlines()
	for sent in examples:
		print(sent)
		parses = parser.parse(sent.split())
		for tree in parses:
			print(tree)

def parse_file(name):
	f = open(name, 'r')
	text = f.read()
	f.close()
	parse_text(text)

print("================ Positive examples ================")
parse_file('positives.txt')
print("================ Negative examples ================")
parse_file('negatives.txt')