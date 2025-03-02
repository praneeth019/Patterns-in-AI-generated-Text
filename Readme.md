# AI Text Detection Using Sequential Pattern Mining

This project implements a machine learning approach to distinguish between AI-generated and human-written text using sequential pattern mining techniques, specifically the SPADE (Sequential Pattern Discovery using Equivalence classes) algorithm.

## Project Overview

The increasing sophistication of AI language models makes it challenging to distinguish between AI-generated and human-written content. This project addresses this challenge by analyzing the sequential patterns in text that might reveal fundamental differences in how AI systems and humans construct sentences.

## Methodology

### Data Processing
1. **Data Balancing**: Equal samples from human and AI sources are selected to prevent bias
2. **Text Preprocessing**: Normalization through lowercasing, punctuation standardization, and tokenization
3. **Train-Test Split**: The dataset is split (70/30) while maintaining class distribution

### Sequential Pattern Mining
1. **Word Indexing**: All unique words are indexed to create a consistent dictionary
2. **Sentence Segmentation**: Text passages are split into sentences
3. **SPADE Algorithm**: Applied separately to AI and human texts to discover frequent sequential patterns
   - Human texts: 0.25 support threshold
   - AI texts: 0.35 support threshold

### Feature Engineering
1. **Sequence Extraction**: Identified frequent sequences from both classes
2. **Feature Vector Creation**: Counts occurrences of each frequent sequence within documents

### Classification Models
Multiple models are trained and evaluated:
1. Logistic Regression
2. Support Vector Machine (Linear)
3. Support Vector Machine (RBF Kernel)
4. Random Forest Classifier

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
spacy
pycspade
```

You'll also need to download the following resources:
- NLTK's punkt tokenizer
- spaCy's English model (en_core_web_sm)

## Usage

1. Prepare your dataset with a 'text' column and a 'source' column (0 for human, 1 for AI)
2. Run the data preprocessing steps
3. Apply the SPADE algorithm to extract sequential patterns
4. Create feature vectors based on sequence frequencies
5. Train and evaluate classification models

## Results

The best performing model is the Random Forest classifier, which demonstrates strong accuracy in distinguishing between AI-generated and human-written text. The sequential patterns discovered through SPADE reveal structural differences in sentence construction between AI and human writing.

## Future Work

1. Experiment with different support thresholds for SPADE
2. Incorporate additional features such as sequence length and part-of-speech patterns
3. Apply more sophisticated ensemble methods
4. Evaluate on different types of AI-generated text to test generalizability

## Conclusion

This project demonstrates that sequential pattern mining can effectively identify differences between AI and human text. The approach focuses on structural patterns rather than content, potentially making it more robust against improvements in AI language models.