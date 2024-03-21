import streamlit as st
import nltk
from nltk.corpus import treebank
from nltk import PCFG, ViterbiParser
nltk.download()
# Load the treebank dataset
# nltk.download('treebank')
corpus = treebank.parsed_sents()

# Train a PCFG parser
productions = []
for tree in corpus:
    productions += tree.productions()
S = nltk.Nonterminal('S')
grammar = nltk.induce_pcfg(S, productions)

# Initialize the parser with the trained grammar
parser = ViterbiParser(grammar)

def evaluate_parser(sentence):
    tokens = nltk.word_tokenize(sentence)
    parsed_trees = list(parser.parse(tokens))
    
    if parsed_trees:
        parsed_tree = parsed_trees[0]
        return parsed_tree
    else:
        return "Failed to parse the sentence."

# Streamlit UI
st.title("PCFG Parser Evaluation")

# Input text box for entering sentences
sentence = st.text_input("Enter a sentence:", "this is a beautiful")

# Button to trigger parsing
if st.button("Parse"):
    parsed_tree = evaluate_parser(sentence)
    st.write("Parsed Tree:")
    st.write(parsed_tree)
