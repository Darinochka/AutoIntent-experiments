# Abstract

# Introduction

Text classification remains a fundamental task in natural language processing, with applications ranging from intent detection in conversational systems to content categorization and sentiment analysis. Modern NLP has been revolutionized by transformer-based embedding models, which provide rich contextual representations of text. However, effectively utilizing these models for classification tasks requires careful consideration of multiple components: the choice of embedding model, the selection of appropriate classification algorithms, and the optimization of their hyperparameters.

Traditional machine learning approaches often require manual tuning of these components, which can be time-consuming and requires significant expertise. While AutoML frameworks have emerged to automate this process, existing solutions for NLP tasks often lack comprehensive support for the full spectrum of hyperparameter optimization, particularly choosing best embedding model and handling multi-label classification and out-of-sample (OOS) detection.

This paper introduces AutoIntent, a novel AutoML framework specifically designed for text classification tasks. AutoIntent addresses these limitations by providing an end-to-end pipeline that automatically optimizes all aspects of the classification process, from embedding model selection to classifier choice and threshold tuning. The framework offers a sklearn-like interface for ease of use while supporting dialog systems-related features such as multi-label classification and OOS handling.

# Background and other automl frameworks

What automl usually looks like, automl for NLP, the limitations of nlp automl frameworks, intent classification challenges

# AutoIntent Library and Framework

## Structure

AutoIntent is designed to solve generic text classification problems with comprehensive support for multi-label classification and out-of-sample (OOS) detection. The framework's architecture is built around a modular design that enables both standalone usage of individual components and end-to-end automated optimization through the AutoML pipeline.

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

3. **BERT-based Classifiers** (GPU-accelerated):
   - Full model fine-tuning
   - Parameter-efficient approaches (LoRA, P-tuning)
   - Cross-encoder configurations for improved accuracy
   These are the only classifiers that require GPU acceleration as they perform end-to-end training.

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
   - **Feature-based**: Assesses embedding models by their performance with a logistic regression classifier. Uses classification metrics like accuracy or ROC AUC, which can also be customized by the user.

2. **Scoring-level Optimization**: Users can opt to optimize the embedding model individually for each scoring module. While this approach might yield better results, it is more time and resource-intensive as it requires refitting and reevaluating each scoring module for every candidate embedding model.

3. **Fixed Embedding**: Users can specify a default embedding model and skip the optimization process entirely, which is useful when the optimal model is known in advance or when computational resources are limited.

The embedding process itself is optimized for efficiency with configurable parameters:
- Batch size for processing multiple texts
- Maximum sequence length for truncation
- Device selection for GPU acceleration
- Model-specific hyperparameters

This flexible approach to embedding model selection allows users to balance between optimization quality and computational efficiency based on their specific needs and resource constraints.

## AutoML Pipeline

The AutoML Pipeline orchestrates the optimization of all components in a sequential manner:

1. **Embedding Optimization**:
   - Evaluates different transformer models
   - Optimizes embedding-specific hyperparameters
   - Uses retrieval hit rate as the optimization metric

2. **Scoring Optimization**:
   - Tests various classification approaches
   - Tunes classifier-specific parameters
   - Optimizes based on ROC AUC score

3. **Decision Optimization**:
   - Configures threshold-based or argmax decision strategies
   - Optimizes for accuracy or other relevant metrics

The pipeline supports multiple hyperparameter optimization strategies:
- Random sampling
- Brute force search
- Tree-structured Parzen Estimators (TPE)

All optimization is implemented using Optuna, providing efficient and flexible hyperparameter tuning. The pipeline can be configured through a declarative search space definition, allowing users to specify:
- Available modules for each stage
- Hyperparameter ranges
- Optimization metrics
- Search strategies

# Experiments

## Scoring Modules (light backbone)

## Scoring Modules (heavy backbone)

## Scoring Modules with different embedding models

## Evolutionary Augmentations

## Simple Augmentations

# AutoIntent Cookbook

# Duscussion and Future Research

# Limitations

# Conclusion
