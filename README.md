# Bayesian Language Model for NLP 使用貝氏定理手刻語言模型

## Overview

This program implements a simple Bayesian language model for text generation in natural language processing. The program is divided into two main parts: `TextProcessor` and `TextGenerator`. `TextProcessor` handles text reading and processing, while `TextGenerator` generates new text based on the processed text.

## Project Summary

This project uses classroom audio recordings as training data to explore the application of handcrafted mathematical models in language processing tasks. By deeply analyzing and extracting features from specific classroom recordings, a series of statistical and rule-based mathematical models were designed. These models cover tasks such as natural language understanding and text generation, and experiments were conducted in different language processing scenarios. The results show that handcrafted mathematical models trained on classroom audio recordings perform well in specific tasks and exhibit competitiveness and practicality in certain situations. The report also analyzes the model's advantages and limitations and provides improvement suggestions.

## Code Structure

### `TextProcessor`

The `TextProcessor` class is responsible for:
- Reading text files.
- Tokenizing text using `jieba`.
- Creating and updating dictionaries to record the next word, previous word, etc.

**Initialization Parameters:**
- `path` (str): The path to the text file to be read.

**Main Methods:**
- `load_text()`: Reads and tokenizes the text.
- `process_text()`: Processes the text and creates various dictionaries.
- `update_dict(target_dict, key, value, is_reverse=False, is_second=False)`: Method to update dictionaries.

### `TextGenerator`

The `TextGenerator` class is responsible for generating text based on the processed text.

**Initialization Parameters:**
- `text_processor` (TextProcessor): An instance of `TextProcessor` used for text processing.
- `start` (str, default '同學'): The starting character for text generation.
- `stop` (str, default '。'): The stopping character for text generation.
- `stop_count` (int, default 3): Maximum steps if the stopping condition is not met.

**Main Methods:**
- `generate_text(max_steps=100000)`: The main method for text generation, where `max_steps` defines the maximum number of steps.
- `step_one()`: Generates the next character based on the current character.
- `step_two()`: Calculates the next character based on probabilities.

## Mathematical Principles

The project solves grammar errors and logical inconsistencies by establishing four types of dictionaries:
- Dictionary for the next character.
- Dictionary for the character two positions ahead.
- Dictionary for the previous character.
- Dictionary for the character two positions before.

Bayes' theorem is used to calculate the probabilities of subsequent characters. The formula for calculating the probability of the next character given certain conditions is:

$$
P(A|B) = \frac{P(A) \cdot P(B|A)}{P(B)}
$$

To calculate the probability of the next character, the formula is:

$$
\text{arg max}_{a \in A} P(a|d_i) = \text{arg max}_{a \in A} P(a) \prod_{j \in N} P(d_i^j | a)
$$

## Experimental Design and Results

Training data is from online audio recordings of AI courses, split into 8941 words, including punctuation, using the `jieba` package. The experimental design includes data collection, text segmentation, dictionary creation, and probability calculation to generate human-like sentences.

## Analysis and Discussion

The study trained a language generation model based on handcrafted mathematical models using classroom audio recordings, achieving good performance in specific tasks. However, the generated text still contains some grammatical errors and logical inconsistencies, requiring further optimization and improvement. The advantages of the model include automated generation of course text, saving preparation time and cost, and generating different course texts based on different teachers' styles and speaking patterns. Challenges include potential grammatical errors and logical inconsistencies in generated text, which require more training data and computational resources. Additionally, the text may lack emotional depth and coherence, needing further enhancement.

## Usage

1. Ensure `jieba` and `numpy` are installed. You can install them using:
    ```bash
    pip install jieba numpy
    ```

2. Prepare a text file, such as `data_set.txt`, and place it in the same directory as the code.

3. Run the program:
    ```bash
    python Bayesian_Language_Model_For_NLP.py
    ```

## Adjustable Parameters

- `path`: The path to the text file. Modify this to the actual file location.
- `start`: The starting character for text generation. Modify as needed.
- `stop`: The stopping character for text generation. Modify as needed.
- `stop_count`: Maximum steps for text generation. Set to `0` to stop when the stopping condition is met.

## Example

Here's an example code snippet:
```python
if __name__ == "__main__":
    path = 'data_set.txt'
    text_processor = TextProcessor(path)
    text_generator = TextGenerator(text_processor)
    text_generator.generate_text()
