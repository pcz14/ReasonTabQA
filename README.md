# ReasonTabQA: A Benchmark for Table Question Answering from Real-World Industrial Scenarios

## Overview

We introduce ReasonTabQA, a comprehensive and industrially-grounded benchmark designed to advance table question answering (Table QA) research under real-world settings. It integrates authentic tables spanning 30 diverse domains and covers four critical table types—single tables, multiple tables, complex structured tables, and extremely large-scale tables. The benchmark is further supported by two specially curated supervised fine-tuning (SFT) datasets that include complete reasoning traces and ground-truth annotations.
To the best of our knowledge, no prior benchmark has been proposed in a form that aligns so closely with genuine industrial scenarios and challenges.

We evaluate 26 competitive baseline methods on ReasonTabQA. Experimental results indicate that even the top-performing model, Doubao-1.5-thinking-pro, attains only 64.48% in overall accuracy. This underscores the substantial challenge posed by real-world industrial Table QA and highlights the need for more robust and specialized solutions.

## The TabCodeRL Framework

We also propose TabCodeRL, a novel two-stage reinforcement learning (RL) framework specifically designed for TableQA. Unlike general-purpose code generation—which often involves multiple programming languages (e.g., Java, C++) and supports a wide variety of solution approaches (where fundamentally different algorithms or logic can correctly solve the same problem)—Table QA primarily operates within the Python/pandas ecosystem. It requires deeper structural comprehension of tabular data and typically admits a more constrained set of correct solutions.

TabCodeRL incorporates a CodeBLEU-based reward function that provides step-level guidance during model uncertainty, reducing ambiguity and delivering structured, semantically-aware feedback. This design makes it particularly effective for handling complex industrial tables with intricate schemas and relationships. To the best of our knowledge, this represents the first integration of a CodeBLEU-based similarity reward within an RL framework for table-based code generation.

## Dataset & Parameter Preview

The ReasonTabQA_samples.zip file contains sample tables from our benchmark, showcasing the four industrially relevant table types. These examples illustrate the challenging nature of table structures and reasoning requirements encountered in practical applications.

The tabcoderl_trainer.yaml file provides key parameter configurations for the reinforcement learning training process within our TabCodeRL framework, serving as a practical reference for model adaptation and evaluation.

## Data & Code Availability

Please note: The ReasonTabQA dataset and the complete codebase will be made publicly available upon the acceptance of our paper. We are committed to supporting reproducible research and will release the comprehensive benchmark and implementation following the review process.
