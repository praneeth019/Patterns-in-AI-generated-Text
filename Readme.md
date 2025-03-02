# Identifying patterns in AI Text Using Multiple Approaches

This project implements machine learning approaches to distinguish between AI-generated and human-written text using three different methods:
1. Basic ML with TF-IDF features
2. Sequential pattern mining (SPADE algorithm)
3. Hierarchical clustering of POS tag n-grams

## Project Overview

The increasing sophistication of AI language models makes it challenging to distinguish between AI-generated and human-written content. This project addresses this challenge through multiple complementary approaches:

1. Using traditional ML models with basic text features
2. Analyzing sequential patterns in text that reveal fundamental differences in how AI systems and humans construct sentences
3. Clustering part-of-speech tag n-grams to identify structural linguistic patterns unique to each source

## Data

**Original source**: https://huggingface.co/datasets/artem9k/ai-text-detection-pile

We reduced this dataset to fit the available compute power. Using the complete dataset will produce better results.

## Methodology

### Common Data Processing
1. **Data Balancing**: Equal samples from human and AI sources are selected to prevent bias
2. **Text Preprocessing**: Normalization through lowercasing, punctuation handling, and tokenization
3. **Train-Test Split**: The dataset is split (70/30) while maintaining class distribution

### Approach 1: Basic ML with TF-IDF
1. **Text Vectorization**: Converting text to numerical features using TF-IDF
2. **Model Training**: Random Forest classifier with hyperparameter optimization
3. **Evaluation**: Accuracy assessment and confusion matrix analysis

### Approach 2: Sequential Pattern Mining
1. **Word Indexing**: All unique words are indexed to create a consistent dictionary
2. **Sentence Segmentation**: Text passages are split into sentences
3. **SPADE Algorithm**: Applied separately to AI and human texts to discover frequent sequential patterns
   - Human texts: 0.25 support threshold
   - AI texts: 0.35 support threshold
4. **Feature Engineering**: Identified frequent sequences are used as features

### Approach 3: Hierarchical Clustering
1. **POS Tagging**: Converting text to part-of-speech tag sequences
2. **N-gram Generation**: Creating n-grams (range 2-3) from POS-tagged text
3. **Word2Vec Embeddings**: Converting n-grams to vector representations
4. **Ward's Method Clustering**: Hierarchical clustering to group similar n-gram patterns
5. **Feature Extraction**: Identifying distinguishing patterns from top and bottom clusters

### Model Optimization
1. **Hyperparameter Tuning**: Using Hyperopt to optimize model parameters
2. **Feature Selection**: Identifying the most discriminative features
3. **Performance Comparison**: Evaluating each approach against baseline models

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
spacy
gensim
pycspade
scipy
hyperopt
```

You'll also need to download the following resources:
- NLTK's punkt tokenizer
- spaCy's English model (en_core_web_sm)

## Usage

1. Prepare your dataset with a 'text' column and a 'source' column (0 for human, 1 for AI)
2. Run the basic ML approach to establish a baseline (mining_ml.ipynb)
3. Apply more advanced techniques:
   - SPADE algorithm for sequential pattern mining
   - Hierarchical clustering for POS tag analysis (Clustering.ipynb)
4. Compare results across different approaches

## Results

The baseline Random Forest model with TF-IDF features provides a strong foundation, but the more advanced techniques offer additional insights:

1. **Basic ML (TF-IDF + Random Forest)**:
   - Quick to implement and provides a solid baseline
   - Performs well but may not capture subtle structural differences

2. **Sequential Pattern Mining**:
   - Reveals distinctive sentence construction patterns
   - Offers interpretable features that show how AI and human writing differ

3. **Hierarchical Clustering**:
   - Identifies linguistic structure differences through POS tag patterns
   - Groups similar patterns to reveal broader writing style differences

The combination of these approaches provides a robust detection system that focuses on different aspects of text generation.

## Future Work

1. Experiment with different feature extraction methods (word embeddings, transformers)
2. Combine all three approaches into an ensemble model
3. Apply more sophisticated optimization techniques
4. Test on different types of AI-generated text to assess generalizability
5. Explore alternative clustering approaches
6. Investigate deeper linguistic features (syntactic dependencies, rhetorical structures)

## Conclusion

This project demonstrates that a multi-faceted approach to AI text detection yields the most comprehensive results. By examining both content and structure through various techniques, we can identify differences between AI and human text that remain consistent even as AI language models continue to improve.
