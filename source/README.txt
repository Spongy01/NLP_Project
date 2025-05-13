############################################
# Team : RAGrats      
# Team Members : Ali Asgar Padaria, Param Patel, Meet Zalavadiya
############################################

This README outlines the correct order for running the files to ensure proper outputs.

We originally stored intermediate outputs in a files/ directory. 
However, due to size constraints (exceeding 1.3GB), these files have been removed. 
Therefore, it is essential to follow the specified execution order.


1. Dataset Preparation
Start by running the notebook in the dataset_creator folder. 
This will load the PubMedQA dataset and create training and validation 
splits of 18,000 and 2,000 rows, respectively—containing approximately 4 million words in total.
The processed datasets will be saved as train_dataset/ and val_dataset/, 
which are used by all subsequent scripts.

2. Train RoBERTa Classifier
Next, train the RoBERTa classifier by running roberta_trainer.ipynb 
inside the roberta_trainer/ folder.
This model is used across all baselines and improvements. 
The trained weights will be saved to the files/ directory and are 
required for downstream evaluations.

3. Run Baselines and Improvements
We recommend running the baseline versions before their corresponding improvements, 
as retrieved contexts are cached in files to avoid redundant computations.

• Baseline 1
Run vectorDB_generator.ipynb to generate embeddings and create a FAISS index.
Then, run baseline_1.ipynb.

• Improvement 1
Simply run improvement_1.ipynb.

• Baseline 2
Run retriever.ipynb to prepare the retrieval module.
Then, run generator.ipynb.

• Improvement 2
Same as Baseline 2: start with retriever.ipynb.
Then, run generator.ipynb.

