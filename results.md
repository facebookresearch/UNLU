# Results

This page contains the results from Table 4 for quick reference. 

- **Original Accuracy**: Acc of the model on the task it has been trained for
- **Max Accuracy**: Maximum accuracy which can be reached by iterative randomization process, which consists of making 100 permutations of each sentence and checking if _any_ permutation yields the correct answer
- **Correct > Random Percentage** : Percentage of examples where models selected > 33% of permutations as true.
- **orig_correct_cor_mean**: Mean number of permutations in 100 permutations which resulted in the correct prediction as of the originally correct prediction
- **flipped_cor_mean**: Mean number of permutations in 100 permutations which resulted in the flip

| Model           | Eval Data   |   Original Accuracy |   Max Accuracy |   Correct > Random Percentage |   orig_correct_cor_mean |   flipped_cor_mean |
|:----------------|:------------|--------------------:|---------------:|------------------------------:|------------------------:|-------------------:|
| RoBERTa (large) | mnli_m_dev  |               0.906 |          0.987 |                         0.794 |                   0.707 |              0.383 |
| RoBERTa (large) | mnli_mm_dev |               0.902 |          0.987 |                         0.79  |                   0.707 |              0.387 |
| RoBERTa (large) | snli_dev    |               0.869 |          0.988 |                         0.826 |                   0.768 |              0.393 |
| RoBERTa (large) | snli_test   |               0.876 |          0.988 |                         0.828 |                   0.76  |              0.407 |
| RoBERTa (large) | anli_r1_dev |               0.458 |          0.897 |                         0.364 |                   0.392 |              0.286 |
| RoBERTa (large) | anli_r2_dev |               0.25  |          0.889 |                         0.359 |                   0.465 |              0.292 |
| RoBERTa (large) | anli_r3_dev |               0.272 |          0.902 |                         0.397 |                   0.48  |              0.308 |
| BART (large)    | mnli_m_dev  |               0.9   |          0.989 |                         0.784 |                   0.689 |              0.393 |
| BART (large)    | mnli_mm_dev |               0.901 |          0.986 |                         0.788 |                   0.695 |              0.399 |
| BART (large)    | snli_dev    |               0.881 |          0.991 |                         0.834 |                   0.762 |              0.363 |
| BART (large)    | snli_test   |               0.879 |          0.99  |                         0.836 |                   0.762 |              0.37  |
| BART (large)    | anli_r1_dev |               0.464 |          0.894 |                         0.374 |                   0.379 |              0.295 |
| BART (large)    | anli_r2_dev |               0.309 |          0.887 |                         0.397 |                   0.428 |              0.303 |
| BART (large)    | anli_r3_dev |               0.327 |          0.931 |                         0.424 |                   0.428 |              0.333 |
| DistilBERT      | mnli_m_dev  |               0.803 |          0.968 |                         0.779 |                   0.775 |              0.343 |
| DistilBERT      | mnli_mm_dev |               0.81  |          0.968 |                         0.786 |                   0.775 |              0.346 |
| DistilBERT      | snli_dev    |               0.738 |          0.956 |                         0.731 |                   0.767 |              0.307 |
| DistilBERT      | snli_test   |               0.739 |          0.95  |                         0.725 |                   0.77  |              0.312 |
| DistilBERT      | anli_r1_dev |               0.237 |          0.75  |                         0.3   |                   0.511 |              0.267 |
| DistilBERT      | anli_r2_dev |               0.272 |          0.76  |                         0.343 |                   0.619 |              0.265 |
| DistilBERT      | anli_r3_dev |               0.311 |          0.83  |                         0.363 |                   0.559 |              0.259 |
| InferSent       | mnli_m_dev  |               0.664 |          0.904 |                         0.712 |                   0.842 |              0.359 |
| InferSent       | mnli_mm_dev |               0.671 |          0.905 |                         0.723 |                   0.844 |              0.368 |
| InferSent       | snli_dev    |               0.549 |          0.82  |                         0.587 |                   0.821 |              0.323 |
| InferSent       | snli_test   |               0.555 |          0.826 |                         0.6   |                   0.824 |              0.321 |
| InferSent       | anli_r1_dev |               0.299 |          0.669 |                         0.313 |                   0.425 |              0.395 |
| InferSent       | anli_r2_dev |               0.292 |          0.662 |                         0.33  |                   0.689 |              0.249 |
| InferSent       | anli_r3_dev |               0.296 |          0.677 |                         0.332 |                   0.675 |              0.236 |
| ConvNet         | mnli_m_dev  |               0.635 |          0.926 |                         0.684 |                   0.773 |              0.34  |
| ConvNet         | mnli_mm_dev |               0.642 |          0.926 |                         0.694 |                   0.782 |              0.343 |
| ConvNet         | snli_dev    |               0.506 |          0.819 |                         0.597 |                   0.813 |              0.339 |
| ConvNet         | snli_test   |               0.494 |          0.821 |                         0.596 |                   0.809 |              0.341 |
| ConvNet         | anli_r1_dev |               0.265 |          0.708 |                         0.316 |                   0.648 |              0.218 |
| ConvNet         | anli_r2_dev |               0.299 |          0.725 |                         0.356 |                   0.703 |              0.224 |
| ConvNet         | anli_r3_dev |               0.319 |          0.798 |                         0.388 |                   0.688 |              0.234 |
| BiLSTM          | mnli_m_dev  |               0.669 |          0.925 |                         0.711 |                   0.8   |              0.351 |
| BiLSTM          | mnli_mm_dev |               0.684 |          0.924 |                         0.724 |                   0.809 |              0.344 |
| BiLSTM          | snli_dev    |               0.536 |          0.86  |                         0.598 |                   0.762 |              0.351 |
| BiLSTM          | snli_test   |               0.539 |          0.862 |                         0.607 |                   0.771 |              0.363 |
| BiLSTM          | anli_r1_dev |               0.261 |          0.671 |                         0.34  |                   0.648 |              0.271 |
| BiLSTM          | anli_r2_dev |               0.298 |          0.728 |                         0.328 |                   0.672 |              0.209 |
| BiLSTM          | anli_r3_dev |               0.292 |          0.731 |                         0.331 |                   0.656 |              0.219 |


