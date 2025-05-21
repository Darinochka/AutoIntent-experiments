# Abstract

# Introduction

Text classification remains a fundamental task in natural language processing, with applications ranging from intent detection in conversational systems to content categorization and sentiment analysis. Modern NLP has been revolutionized by transformer-based embedding models, which provide rich contextual representations of text. However, effectively utilizing these models for classification tasks requires careful consideration of multiple components: the choice of embedding model, the selection of appropriate classification algorithms, and the optimization of their hyperparameters.

Traditional machine learning approaches often require manual tuning of these components, which can be time-consuming and requires significant expertise. While AutoML frameworks have emerged to automate this process, existing solutions for NLP tasks often lack comprehensive support for the full spectrum of hyperparameter optimization, particularly choosing best embedding model and handling multi-label classification and out-of-sample (OOS) detection.

This paper introduces AutoIntent, a novel AutoML framework specifically designed for text classification tasks. AutoIntent addresses these limitations by providing an end-to-end pipeline that automatically optimizes all aspects of the classification process, from embedding model selection to classifier choice and threshold tuning. The framework offers a sklearn-like interface for ease of use while supporting dialog systems-related features such as multi-label classification and OOS handling.

# Background and other automl frameworks

AutoML, by definition, is a tool for automating machine learning routines, including data splitting for validation, hyperparameter tuning, model ensembling, and model selection experiments. Different domains of AutoML have their specific challenges and solutions:

- **Tabular AutoML**: Focuses on feature engineering and feature selection
- **Deep Learning AutoML**: Emphasizes architecture search and occasianally data augmentations
- **NLP AutoML**: Primarily revolves around training recipes for transformer-based models

Current NLP AutoML frameworks can be categorized based on their approaches:

1. **Transformer-focused Frameworks**: Most existing solutions concentrate on optimizing BERT-like models, often neglecting other aspects of the ML pipeline.

2. **Meta-learning Approaches**: Some frameworks, like Text-Brew, incorporate meta-learning techniques to improve model selection and optimization.

3. **Budget-aware Frameworks**: Advanced solutions include granular optimization budget control, allowing users to balance between performance and computational resources.

However, a critical analysis of existing NLP AutoML frameworks reveals several limitations:

1. **Incomplete Pipeline Coverage**: Most frameworks do not consider the full ML pipeline, particularly:
   - Data splitting strategies for validation
   - Threshold tuning for classification
   - Native support for out-of-scope (OOS) detection

2. **Limited Text Processing Options**: As shown in Table 1, current frameworks offer limited text processing capabilities:
   - Some rely solely on embeddings
   - Others use basic TF-IDF approaches
   - Few support advanced techniques like prompt engineering

3. **Resource Constraints**: Many frameworks are not optimized for:
   - Small dataset scenarios
   - Limited computational resources
   - Efficient hyperparameter optimization

Table 1: Comparison of NLP AutoML frameworks
| Feature | H2O | LightAutoML | AutoGluon | FEDOT |
|---------|-----|-------------|-----------|-------|
| Text Processing | No built-in support | TF-IDF and embeddings | Embeddings only | TF-IDF, embeddings |
| Small Data Support | Not optimized | Has small data modes | No support | Adaptable to small data |
| Parameter Configuration | Flexible API | Limited preset configs | Custom configs, poor docs | Limited configuration |
| External Logging | H2O Flow integration | No support | No support | No support |
| Encoder Prompts | No support | No support | No support | No support |
| OOS Detection | No built-in support | No built-in support | No support | No support |

These limitations highlight the need for a more comprehensive NLP AutoML framework that addresses the full spectrum of text classification challenges, from data preprocessing to model deployment, while maintaining flexibility and efficiency across different resource constraints.

# AutoIntent Library and Framework

## Structure

AutoIntent is designed to solve generic text classification problems with comprehensive support for multi-label classification and out-of-scope (OOS) detection. The framework's architecture is built around a modular design that enables both standalone usage of individual components and end-to-end automated optimization through the AutoML pipeline.

The framework's key features include:

- **Sklearn-like Interface**: All components follow the familiar sklearn estimator interface, making the framework accessible to users familiar with standard machine learning workflows. This includes consistent fit/predict methods and compatibility with sklearn's cross-validation and evaluation tools.

- **AutoML Pipeline**: The core of AutoIntent is its automated optimization pipeline, which sequentially optimizes three main components:
  1. Embedding Module: Selects and optimizes the most appropriate transformer model from Hugging Face for the given task
  2. Scoring Module: Identifies and tunes the best performing classifier for the task
  3. Decision Module: Optimizes classification thresholds for multi-label scenarios and OOS detection

- **Augmentation Modules**: The framework includes advanced data augmentation capabilities:
  - LLM-based augmentation for generating synthetic training examples
  - Evolutionary approach for transforming existing utterances to expand the training set
  - Customizable prompts to fine-tune the augmentation process for your needs

## Separation of Concerns

In AutoIntent, a scoring module is defined as a model that takes a text sample as input and outputs a vector of class probabilities. This definition is crucial as it establishes a clear separation of concerns in the framework's architecture: scoring modules are responsible for probability estimation, while decision modules take these probabilities as input and produce the final classification decisions (single label, multiple labels, or OOS detection).

This separation of concerns provides several key benefits:

1. **Modularity**: Each component can be developed, tested, and optimized independently. Scoring modules focus solely on learning the relationship between text and class probabilities, while decision modules concentrate on the optimal way to convert these probabilities into final predictions.

2. **Flexibility**: The same scoring module can be paired with different decision modules to address various scenarios. For example, a single scoring module can be used with:
   - A threshold-based decision module for binary classification
   - An adaptive decision module for multi-label classification
   - A specialized OOS detection decision module

3. **Reusability**: Probability estimates from scoring modules can be stored and reused with different decision strategies without recomputing the scores, introducing AutoIntent's efficient experimentation with various decision-making approaches.

4. **Interpretability**: The separation allows for better understanding of the model's behavior, as probability estimates can be analyzed independently of the final decision-making process.

## Scoring Modules

AutoIntent offers a diverse set of scoring modules, each implementing different classification approaches. A key architectural feature of the framework is that all classifiers (except BERT-based ones) operate on pre-computed embeddings as features. This design choice provides several advantages:

1. **Efficiency**: By separating the computationally intensive embedding computation from classification, the framework achieves a fine balance between effectiveness and computational efficiency. The embedding vectors serve as rich, pre-computed features that capture the semantic meaning of the text, while the classifiers themselves are lightweight and fast to train and deploy.

2. **Hardware Flexibility**: This architecture allows the framework to be deployed on a broad range of machines, even without GPU acceleration. The embedding computation can be done once and cached, while the classification models can run efficiently on CPU-only systems.

The available scoring modules include:

1. **Linear Classifier**: Utilizes sklearn's LogisticRegressionCV for efficient linear classification with built-in cross-validation. The linear model operates directly on the embedding vectors, providing fast training and inference times.

2. **KNN-based Approaches**:
   - Standard KNN with FAISS for efficient nearest neighbor search in the embedding space
   - Rerank: A two-stage approach combining retrieval and reranking, both operating on pre-computed embeddings
   - MLKNN: A specialized multi-label KNN implementation with Bayesian background, leveraging the embedding space for similarity calculations

3. **BERT-based Classifiers**:
   - Full model fine-tuning
   - Parameter-efficient approaches (LoRA, P-tuning).

4. **Generic sklearn Integration**: Support for any sklearn classifier, allowing for easy extension with additional classification algorithms. All sklearn models operate on the embedding vectors as features, maintaining the efficiency benefits of the architecture.

Each scoring module follows a consistent interface, providing both standard prediction methods and rich output options for detailed analysis and debugging.

## Decision Modules

The Decision Module handles the final classification decisions, particularly crucial for multi-label scenarios and out-of-sample detection. It processes the probability scores from the Scoring Module to produce the final set of predicted labels. AutoIntent provides several specialized decision strategies:

1. **AdaptiveDecision**: A sample-specific thresholding approach that calculates thresholds tailored for each text sample based on the minimum and maximum class scores. This method is specifically designed for multi-label classification scenarios, where it can better handle the varying confidence levels across different samples. However, it does not support out-of-sample detection.

2. **JinoosDecision**: A specialized approach for handling out-of-sample (OOS) detection, adapted from the DNNc framework. This method finds a universal threshold that balances between in-domain and out-of-domain accuracy, making it particularly suitable for applications where detecting unknown intents is crucial.

3. **ThresholdDecision**: A simple but effective approach using a fixed threshold specified during model initialization. This method provides a straightforward way to control the classification confidence level and can be useful in scenarios where consistent decision boundaries are required. Also this module is handy to use within AutoML pipeline with optuna samplers, which we will discuss later.

4. **TunableDecision**: An automated threshold optimization approach that uses Optuna to find the optimal threshold value by maximizing the F1 score on the given data. This method is particularly useful when the optimal threshold is not known in advance and needs to be determined empirically.

## Embedding Modules

The Embedding Module is responsible for converting text inputs into vector representations using transformer models. AutoIntent leverages the sentence-transformers library, providing access to a wide range of pre-trained transformer models from Hugging Face. While the module can be used standalone, its primary role is within the AutoML pipeline, where it serves as a single-time optimization step to select the most appropriate embedding model for the task.

The framework offers three strategies for handling embedding model selection:

1. **Pipeline-level Optimization**: The embedding model is chosen once at the pipeline level, before proceeding to classifier optimization. This approach is more efficient because it avoids the need to refit and reevaluate each scoring module for every candidate embedding model. Users can select from two optimization criteria:
   - **Retrieval-based**: Evaluates embedding models based on their ability to match samples within the same class. Quality is measured using retrieval metrics such as hit rate or NDCG, which can be customized by the user.
   - **Classification-based**: Assesses embedding models by their performance with a logistic regression classifier. Uses classification metrics like accuracy or ROC AUC, which can also be customized by the user.

2. **Scoring-level Optimization**: Users can opt to optimize the embedding model individually for each scoring module. While this approach might yield better results, it is more time and resource-intensive as it requires refitting and reevaluating each scoring module for every candidate embedding model.

3. **Fixed Embedding**: Users can specify a default embedding model and skip the optimization process entirely, which is useful when the optimal model is known in advance or when computational resources are limited.

The embedding process itself is optimized for efficiency with configurable parameters:
- Batch size for processing multiple texts
- Maximum sequence length for truncation
- Device selection for GPU acceleration
- Model-specific hyperparameters

This flexible approach to embedding model selection allows users to balance between optimization quality and computational efficiency based on their specific needs and resource constraints.

## AutoML Pipeline

The AutoML Pipeline orchestrates the optimization of all components in a hierarchical manner, with three distinct levels of optimization:

1. **Modules-level Optimization**:
   - Sequential optimization of embedding, scoring, and decision modules
   - Each module uses the best model from the previous module's optimization
   - Example: scoring module uses features from the best embedding model, decision module uses probabilities from the best classifier
   - This greedy approach prevents combinatorial explosion while maintaining good performance

2. **Model-level Optimization**:
   - Selection of the best model type for each module
   - Exhaustive evaluation of all models specified in the search space
   - Examples:
     - Embedding module: different transformer models
     - Scoring module: KNN, Logistic Regression, BERT fine-tuning, sklearn classifiers
     - Decision module: different threshold strategies

3. **Hyperparameter-level Optimization**:
   - Tuning of model-specific hyperparameters
   - Optimization of trainable weights where applicable
   - Examples:
     - Tuning: K for KNN, threshold values
     - Training: feature weights in Logistic Regression
   - Uses Optuna with three sampling strategies:
     - Random sampling
     - Brute force search
     - Tree-structured Parzen Estimators (TPE)

   It's important to note that AutoIntent makes a clear distinction between tuning and training:
   - **Tuning** refers to the optimization of hyperparameters that control the model's behavior (e.g., number of neighbors in KNN)
   - **Training** refers to the optimization of model weights that are learned from the data (e.g., coefficients in Logistic Regression)
   This distinction is crucial for proper model optimization and evaluation.

The pipeline employs careful data handling to prevent target leakage and ensure reliable model selection:

- **Training Data**: Used for training model weights and parameters
- **Validation Data**: Used for hyperparameter tuning and model selection
- **Separate Validation Sets**: Scoring and decision modules can use different validation and training splits to prevent target leakage during holdout scoring

The optimization process is configured through a declarative search space definition, allowing users to specify:
- Available modules for each stage
- Hyperparameter ranges
- Optimization metrics
- Search strategies

The result of the AutoML pipeline is a fully optimized classification system that includes:
- The best embedding model for the task
- The optimal scoring module with tuned hyperparameters and trained weights
- The appropriate decision module with optimized thresholds
- All necessary configuration for inference using intuitive `save`, `load` and `predict` methods

This hierarchical approach ensures efficient optimization while maintaining the flexibility to explore different model combinations and configurations.

# Experiments

## Scoring Modules (light backbone)

We conducted a comprehensive evaluation of all scoring models across a range of popular intent classification datasets using lightweight backbone and embedding models. The experimental setup employed holdout validation (see dataset statistics in Appendix TODO) and hyperparameter optimization using the TPE sampler with a maximum of 20 trials (see complete search space specifications in Appendix TODO).

The results, presented in Table 1, reveal several key findings:

1. Logistic regression ("linear" in the table) demonstrates superior performance in terms of both average accuracy and consistency across datasets, achieving the highest score in three out of five datasets.

2. BERT-based methods exhibit significant variability in performance:
   - Parameter-efficient approaches (P-tuning and LoRA) show notably lower performance
   - Full BERT fine-tuning achieves competitive results in one dataset but underperforms in others
   - This suggests potential challenges in hyperparameter optimization for transformer-based methods

3. Feature-based methods (linear, KNN, random forest) consistently outperform transformer-based approaches, with all three achieving average accuracy above 90%.

```
Table 1: Performance of scoring modules with lightweight backbone
 model name  banking77  minds14  hwu64  snips  massive  average  best_count
    ptuning       4.63    11.63   3.65  66.49     8.57    18.99           0
       lora      20.35    12.40  22.40  95.09    41.95    38.44           0
       bert      64.14    69.77  73.40  98.52    76.76    76.52           1
     rerank      89.04    97.67  84.45  96.59    81.68    89.89           0
 sklearn rf      89.81    98.45  86.98  95.16    79.73    90.03           2
        knn      89.74    97.67  85.42  96.10    81.65    90.12           0
     linear      90.51    97.67  89.17  97.45    84.70    91.90           3
```

TODO: research the overfitting of bert-based models

## Scoring Modules (heavy backbone)

To investigate the impact of model capacity on performance, we repeated the experiment using heavier transformer models. While the overall ranking by average accuracy remained consistent with the lightweight experiment, the distribution of best-performing models across datasets shifted:

1. The dominance of logistic regression became less pronounced, with KNN and random forest achieving comparable performance in terms of dataset-specific wins.

2. BERT-based methods continued to underperform relative to feature-based approaches, though full BERT fine-tuning showed improved results in some datasets. This suggests that increased model capacity alone may not address the optimization challenges of transformer-based methods.

3. The competitive performance of multiple feature-based methods justifies the inclusion of diverse scoring modules in the library, particularly for applications where identifying the optimal model is critical.

```
Table 2: Performance of scoring modules with heavy backbone
 model name   banking77 minds14  hwu64  snips  massive  average  best_count
    ptuning       5.00    14.73   9.64  91.67    25.87    29.38           0
       lora      13.82     9.30  38.58  99.13    52.06    42.58           0
       bert      85.61    48.06  88.58  99.16    86.36    81.56           1
     rerank      90.28    96.12  87.24  97.35    83.87    90.97           0
 sklearn rf      91.54    98.45  90.22  96.64    85.08    92.39           2
        knn      92.14    98.45  89.43  96.97    85.43    92.48           2
     linear      91.81    96.90  91.78  98.06    87.58    93.23           2
```

## Scoring Modules with different embedding models

## Computational Efficiency

To quantify the computational requirements of different scoring modules, we conducted a comprehensive analysis using the Code Carbon library. This analysis measured various aspects of computational resource consumption for a single trial (training and evaluation of a single model configuration). The results, presented in Table 3, reveal significant variations in resource usage across different approaches.

Table 3: Computational resource consumption for different scoring modules. The experiments are conducted on banking77 dataset with mixedbread-ai/mxbai-embed-large-v1, system with AMD Ryzen 7 5800H, NVIDIA RTX 3060 Laptop. Median values of 10 trials are displayed. Embeddings were pre-computed.

Measurements:
- emissions: CO2 in grams
- runtime: seconds
- energy_consumed, gpu_energy, cpu_energy, ram_energy: watt-hours
- emissions_rate: CO2 grams per second

```
scorer_name  emissions   runtime  energy_consumed    gpu_energy  cpu_energy  ram_energy  emissions_rate
        bert      1.382   103.911            3.133         2.198       0.774   1.615e-01           0.014
     ptuning      1.118    83.455            2.535         1.785       0.620   1.295e-01           0.014
        lora      0.863    65.157            1.957         1.372       0.484   1.009e-01           0.013
      linear      0.428    73.393            0.971         0.312       0.545   1.138e-01           0.006
      rerank      0.270    29.040            0.613         0.355       0.213   4.436e-02           0.010
        dnnc      0.122    10.000            0.276         0.192       0.070   1.455e-02           0.013
 rand forest      0.073    11.367            0.166         0.074       0.080   1.664e-02           0.007
         knn      0.009     1.281            0.019         0.014       0.004   9.044e-04           0.012
```

Key findings from the analysis:

1. **Resource Efficiency Ranking**:
   - KNN-based methods demonstrate exceptional efficiency, with minimal emissions and runtime
   - Logistic regression shows moderate resource consumption, correlating with its high performance
   - BERT-based methods exhibit the highest resource requirements

2. **Energy Consumption Patterns**:
   - GPU utilization varies significantly across methods:
     - BERT-based methods maintain high GPU power (76-77W)
     - Linear methods show lower GPU usage (15.46W)
     - KNN methods, despite low total energy, show high GPU power utilization
   - CPU power remains consistent across all methods (27W)
   - RAM energy consumption is highest for BERT-based methods

These findings highlight the trade-off between computational efficiency and model performance, providing valuable insights for resource-constrained deployment scenarios.

## Evolutionary Augmentations

To evaluate the effectiveness of LLM-based data augmentation in real-world scenarios with limited training data, we conducted our experiments on randomly sampled 10-shot versions of the datasets. This setup simulates common practical situations where end users need to expand their small training splits. We conducted a comprehensive correlation analysis between the number of synthetic samples added and the final classification accuracy. We tested four state-of-the-art LLMs (DeepSeek-V3-0324, GPT-4o-mini, Qwen2.5-7B, and Llama-3.1-8B) across multiple datasets, including English (banking77, snips), Russian (banking77_ru, snips_ru), and out-of-sample (clinc150_ru) datasets.

The correlation analysis revealed several key findings:

**Model-specific Patterns**. Surprisingly, the size and type of the LLM (proprietary vs. open-source) showed no clear correlation with augmentation effectiveness. Both smaller models like Llama-3.1-8B and larger models like DeepSeek-V3-0324 demonstrated similar patterns of effectiveness.

These results suggest that:
1. The effectiveness of augmentation doesn't vary significantly across different LLMs, but with GPT-4o-mini showing the most consistent positive impact
2. English datasets generally benefit more from augmentation than their Russian counterparts, likely due to the English-oriented nature of the prompts we used for augmentation
3. OOS detection performance tends to decrease with increased augmentation, particularly for certain models. This can be attributed to the combination of factors: the Russian language of the dataset, its high granularity (150 classes), and the generic nature of synthetic samples that fail to capture the fine-grained class boundaries
4. The banking77 dataset shows the most consistent improvement with augmentation across all models

comparison 1:
```
Correlation Analysis (r values, non-significant correlations omitted):
                           Model English Russian    OOS
                DeepSeek-V3-0324   0.532       - -0.777
          gpt-4o-mini-2024-07-18   0.500       -  0.884
         Qwen2.5-7B-Instruct-AWQ       -       - -0.613
Meta-Llama-3.1-8B-Instruct-Turbo   0.568       - -0.932
```

comparison 2:
```
Dataset-specific correlations across all models:
     Dataset Correlation
    snips_ru           -
       snips       0.459
 clinc150_ru           -
banking77_ru       0.315
   banking77       0.728
```

To further investigate the impact of augmentation on model performance, we conducted a detailed statistical analysis comparing the accuracy before and after augmentation for each augmentation level (1-10 synthetic samples). We performed paired t-tests to assess the statistical significance of the improvements and calculated effect sizes using Cohen's d to quantify the magnitude of the changes.

The analysis revealed that all augmentation levels resulted in statistically significant improvements (p < 0.01) compared to the baseline (no augmentation). The effect sizes ranged from 0.33 to 0.57, indicating moderate to large practical significance (cohen's d). The mean improvement in accuracy showed a generally increasing trend with the number of augmentations, ranging from 7.1% to 11.7% improvement over the baseline.

These results demonstrate that:
1. Any amount of augmentation (1-10 samples) leads to statistically significant improvements in model performance
2. The relationship between the number of augmentations and performance improvement is not strictly linear, with quite early saturation
3. The improvements are not only statistically significant but also practically meaningful, as indicated by the moderate to large effect sizes

This analysis provides strong empirical support for the effectiveness of LLM-based augmentation in improving classification performance, while also suggesting optimal ranges for the number of synthetic samples to generate.

```
Before and after analysis results:
[
  {
    "naug": 1,
    "t_stat": 3.735546060278238,
    "pval": 0.0018019139464879965,
    "effect_size": 0.35508506925428296,
    "mean_improvement": 0.07141329258976314,
    "n_comparisons": 17
  },
  {
    "naug": 2,
    "t_stat": 8.398754713510861,
    "pval": 2.933752247838802e-07,
    "effect_size": 0.366245426026366,
    "mean_improvement": 0.07244079449961793,
    "n_comparisons": 17
  },
  {
    "naug": 3,
    "t_stat": 3.2798027445708984,
    "pval": 0.004715696559629987,
    "effect_size": 0.33069460006391044,
    "mean_improvement": 0.07844155844155842,
    "n_comparisons": 17
  },
  {
    "naug": 4,
    "t_stat": 3.875191988007287,
    "pval": 0.0013421265922781013,
    "effect_size": 0.3653368491467401,
    "mean_improvement": 0.08334606569900682,
    "n_comparisons": 17
  },
  {
    "naug": 5,
    "t_stat": 3.2470128092719794,
    "pval": 0.0050526718294561825,
    "effect_size": 0.3509996686289342,
    "mean_improvement": 0.08577158135981666,
    "n_comparisons": 17
  },
  {
    "naug": 6,
    "t_stat": 4.990219384905349,
    "pval": 0.00013347561041411162,
    "effect_size": 0.4083321024424291,
    "mean_improvement": 0.08789152024446145,
    "n_comparisons": 17
  },
  {
    "naug": 7,
    "t_stat": 6.751435624288144,
    "pval": 4.651938073069818e-06,
    "effect_size": 0.4703528563923851,
    "mean_improvement": 0.10019480519480517,
    "n_comparisons": 17
  },
  {
    "naug": 8,
    "t_stat": 7.3885984719419096,
    "pval": 1.5311770165757748e-06,
    "effect_size": 0.5670558638421866,
    "mean_improvement": 0.11709320091673026,
    "n_comparisons": 17
  },
  {
    "naug": 9,
    "t_stat": 4.088494005336879,
    "pval": 0.0008569900951346268,
    "effect_size": 0.38933898033106124,
    "mean_improvement": 0.08583651642475154,
    "n_comparisons": 17
  },
  {
    "naug": 10,
    "t_stat": 5.632600136036402,
    "pval": 3.742690319318556e-05,
    "effect_size": 0.4056996407946527,
    "mean_improvement": 0.0872650878533231,
    "n_comparisons": 17
  }
]
```

## Baselines

we have tested some opensource automl nlp frameworks as a baselines;

- h2o peforms tabular automl methods over word2vec embeddings
- lightautoml (lama) and fedot perform automl over tf-idf, though lama can use feature of one of three predefined transformers 
- autogluon trains deberta-v3-small using pytorch lightning

```
framework  banking77  clinc150   hwu64  massive  minds14  snips  average
autointent     92.86             90.83    87.13    95.68  98.19
autogluon      93.28     87.35   91.17    88.92    97.22  99.07    92.83
     h2o       75.32     66.31   77.32    75.30    76.85  98.36    78.24
   fedot        1.30     18.18    1.77     7.28    12.04  15.14     9.28
    lama        1.30     18.18    1.77     7.04     8.33  14.50     8.52
```

autointent results are taken from experiment on scoring modules with heavy backbone

the results are that two of four frameworks completely failed to train a model. it marks their drawback to adapt to given data. most probably it is explained by the fact that they do not tune hyperparameters. the most obvious drawbacks of the frameworks are the following:
- gluon does not tune hyperparameters and it uses fixed training recipe
- h2o and fedot doesnt support features from transformer 
- lama do support features from transformer but it doesnt allow to set any transformer from hugging face hub you want
- gluon always outputs fixed inference time model while autointent can choose lighter model is its perform on par or better that heavy fine-tune-based approaches
- (?) we didnt find mature deployment features; in contrast autointent provides straightforward method to save to disk and load models

## AutoIntent Cookbook

Based on our extensive experimental analysis, we present a comprehensive guide for selecting and configuring AutoIntent components for different use cases. To facilitate the adoption of these recommendations, we have implemented them as configurable presets in our library.

## Resource-aware Configuration Presets

We have identified three distinct configuration presets that balance performance and computational requirements:

1. **Lightweight Configuration**:
   - **Components**: KNN-based methods
   - **Use Case**: 
     - Scenarios requiring minimal resource consumption
     - Few-shot learning scenarios with limited training data
   - **Advantages**:
     - Fast inference time
     - Minimal computational requirements for training
     - Particularly effective in few-shot learning scenarios
   - **Performance**: 
     - Achieves competitive accuracy with significantly reduced resource usage
     - Demonstrates strong performance on small datasets, often outperforming more complex models in few-shot settings
     - Maintains consistent performance across different data sizes

2. **Moderate Configuration**:
   - **Components**: Linear scorer, cross-encoder based methods, and all lightweight methods
   - **Use Case**: Applications requiring consistent, high-quality performance
   - **Advantages**:
     - Balanced performance and resource usage
     - Robust across different datasets
     - Suitable for production environments
   - **Performance**: Provides reliable accuracy with moderate computational overhead

3. **Heavy Configuration**:
   - **Components**: BERT-based models, along with all moderate and lightweight methods
   - **Use Case**: High-performance scenarios with access to professional-grade GPUs
   - **Requirements**:
     - High-performance GPU with substantial VRAM
     - Extended training time
     - Higher computational budget
     - Large dataset
   - **Performance**: Potential for superior accuracy with heavy backbones and longer training

## Selection Guidelines

When choosing a configuration preset, consider the following factors:

1. **Resource Constraints**:
   - Available computational resources
   - Inference time requirements
   - Memory limitations

2. **Performance Requirements**:
   - Required accuracy
   - Latency constraints
   - Scalability needs

3. **Dataset Characteristics**:
   - Size of the training data
   - Complexity of the classification task
   - Language and domain specifics

## Implementation Notes

Each preset is implemented as a configurable template in the library, allowing users to:
- Start with recommended configurations
- Customize parameters based on specific needs
- Gradually scale up from lightweight to heavy configurations as requirements evolve

This cookbook approach enables users to quickly deploy effective solutions while maintaining the flexibility to adapt to changing requirements and resource constraints.

# Future Research

Our experimental analysis and framework development have revealed several promising directions for future research:

1. **Framework Evaluation Extensions**:
   - Comprehensive evaluation of module data separation strategies
   - In-depth analysis of multi-label classification performance
   - Systematic assessment of out-of-scope (OOS) detection capabilities
   - Investigation of cross-lingual transfer capabilities

2. **Augmentation Method Improvements**:
   - Development of more sophisticated prompt engineering techniques
   - Integration of domain-specific knowledge into augmentation strategies
   - Exploration of hybrid approaches combining LLM-based and traditional augmentation methods
   - Investigation of augmentation quality control mechanisms
   - Research into language-specific augmentation strategies

3. **BERT-based Methods Enhancement**:
   - Evaluation of transformer models on high-performance GPUs (e.g., A100)
   - Investigation of potential overfitting issues in current implementations
   - Development of more efficient fine-tuning strategies
   - Exploration of model compression techniques for deployment

4. **Pipeline Optimization**:
   - Development of more sophisticated module selection strategies
   - Investigation of joint optimization approaches
   - Research into adaptive resource allocation during training
   - Exploration of automated pipeline configuration based on dataset characteristics

5. **Deployment and Scalability**:
   - Research into efficient model serving strategies
   - Investigation of distributed training approaches
   - Development of automated scaling mechanisms
   - Exploration of edge deployment capabilities

These research directions aim to address current limitations and expand the framework's capabilities, making it more robust and adaptable to various real-world scenarios.

# Limitations

# Conclusion
