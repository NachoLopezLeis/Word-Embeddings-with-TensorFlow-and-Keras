# Word Embeddings with TensorFlow and Keras

This project demonstrates how to build, train, and analyze word embedding models using TensorFlow and Keras. The workflow includes data preprocessing, model definition, training, and the evaluation of embeddings through similarity analysis.

---

## **Project Overview**

- **Goal:**  
  Generate and compare word embeddings from a text dataset using custom neural network models, and analyze their semantic properties via cosine similarity matrices.

- **Main Steps:**  
  - Data loading and preprocessing  
  - Building various embedding models with different dimensions and window sizes  
  - Training the models  
  - Extracting and visualizing word embeddings  
  - Analyzing word similarity using cosine similarity matrices

---

## **Dataset**

- The code expects two CSV files: `train.csv` and `test.csv`.
- The training dataset (`train.csv`) includes columns such as:
  - `textID`
  - `text`
  - `selected_text`
  - `sentiment`
  - `Time of Tweet`
  - `Age of User`
  - `Country`
  - `Population -2020`
  - `Land Area (Km²)`
  - `Density (P/Km²)`

---

## **Main Dependencies**

- Python 3.x
- numpy
- pandas
- tensorflow (keras)
- scikit-learn
- matplotlib, seaborn
- spacy

---

## **How It Works**

### **1. Data Preprocessing**

- Loads the training and test datasets.
- Cleans text data using regular expressions (removes punctuation, converts to lowercase).
- Tokenizes text and prepares sequences for training skip-gram models.

### **2. Model Architecture**

- Multiple embedding models are defined, varying in embedding size (e.g., 45, 312, 752) and context window size (e.g., 2, 4).
- Each model uses:
  - Two input layers (target word, context word)
  - Embedding layers for both target and context
  - Dot product to compute similarity
  - Dense layer for output

### **3. Training**

- Models are trained using the skip-gram approach, aiming to predict context words given a target word.
- Training is performed for each combination of embedding size and window size.

### **4. Embedding Extraction and Analysis**

- After training, embeddings are extracted for each word.
- Cosine similarity matrices are computed for selected words to analyze semantic relationships.
- Results are visualized using heatmaps and printed matrices.

---

## **Example Outputs**

- **Model Summaries:**  
  The code prints detailed model summaries, including the number of parameters for each architecture.

- **Embedding Vectors:**  
  Example:  
Embedding for 'love':
[ 0.03099482 -0.03489054 ... -0.17953782]


- **Cosine Similarity Matrices:**  
The code outputs similarity matrices for a set of words (e.g., "love", "day", "night", etc.), helping to evaluate the quality of the learned embeddings.

---

## **Usage**

1. Place `train.csv` and `test.csv` in the working directory.
2. Run the notebook `PW2_embedddings_1.ipynb` in a Jupyter environment or Google Colab.
3. Adjust embedding sizes and window sizes as needed by modifying the relevant sections.
4. Explore the generated embeddings and similarity matrices.

---

## **Customization**

- You can change the set of words for similarity analysis by editing the relevant code section.
- The embedding size and window size can be tuned to experiment with different semantic properties.

---

## **References**

- The implementation is inspired by standard word2vec skip-gram models and extends them for custom analysis.

