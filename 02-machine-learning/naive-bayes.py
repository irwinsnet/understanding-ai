import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to Machine Learning
    ### A Naive Bayes Text Message Classifer

    In this exercise, we're going to learn basic principles of machine learning by building a text message classifier. There are many different kinds of machine learning models. We're going to use a naive Bayes (NB) model.

    Why are we talking about Naive Bayes models when our goal is to understand how LLMs work? The workings of LLMs is sufficiently complicted that you have to learn a lot of other things first. NB models and text message clasification is a good first step.
    * Classifiying text messages will introduce us to a few principles of natural language processing (NLP). (LLMs are highly advanced natural language processors.)
    * NB models are easy to program and it's easy to understand how they work.
    * NB models work well with small datasets.
    * NB models and LLMs are both machine learning models. Working with NB models will help us learn some basic principles of machine learning (training, evaluation metrics, parameters, etc.) that also apply to LLMs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    Here are the packages we'll use in this notebook.
    """)
    return


@app.cell
def _():
    # Imports from the Python Standard Library
    import dataclasses
    import math
    import pathlib
    import random

    # Third Party Packages
    # Used in all Marimo notebooks
    import marimo as mo
    # Natural Language Toolkit
    import nltk

    return dataclasses, math, mo, nltk, pathlib, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `nltk` package is the [Natural Language Toolkit](https://pypi.org/project/nltk/). NLTK has been around since 2001, longer than AI, smartphones, and YouTube. It does a lot of interesting things. [Check out the NLTK book to learn more.](https://www.nltk.org/book)

    /// attention
    If you have not yet done so, follow the instructions in the README.md file to run the `nltk.download()` command in your terminal. The `nltk.download()` command won't work from a Marimo notebook.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Dataset
    When building a machine learning model, the first thing you need is a dataset. We're going to use the *SMS SPAM Collection* which is available at the [University of California Irvine (UCI) Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). The UCI repository is a good source for datasets if you want to experiment with machine learning algorithms. FYI, SMS stands for *Simple Message Service*, which is the technical term for text messaging over cellular networks.

    Let's load the dataset and print the first 1000 characters.
    """)
    return


@app.cell
def _(pathlib):
    DATA_PATH = pathlib.Path.cwd() / "02-machine-learning" / "data"

    def load_sms_text() -> str:
        """Load the SMS text message dataset.

        Returns:
            A string containing the entire dataset.
        """
        with open(DATA_PATH / "SMSSpamCollection.txt") as textfile:
            return textfile.read()

    sms_text = load_sms_text()
    return (sms_text,)


@app.cell
def _(sms_text):
    print(sms_text[:1000])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The SMS dataset consists of 5,574 text messages.
    * There are 450 spam messages in the dataset, and the remaining 5,124 messages are not spam.
    * Each line in the file contains one text message.
    * The first word on each line is either "spam" or "ham." *Ham* means the text message isn't spam.

    Let's convert the text to a list.
    """)
    return


@app.cell
def _(sms_text):
    def build_dataset(text: str) -> list[tuple[str, str]]:
        """Convert the dataset string into a list of messages..

        Args:
            text: The entire SMS dataset as a string.

        Returns:
            A list of 2-item tuples.
            * The first item is "spam" for spam messages or "ham"
              for non-spam messages.
            * The second item is the text message.
        """
        lines = sms_text.splitlines()
        dataset = []
        for line in lines:
            words = line.split()
            category = words[0]
            assert category in ["spam", "ham"]  # This should be true, but raise an error if not.
            message = " ".join(words[1:])
            dataset.append((category, message))
        return dataset

    dataset = build_dataset(sms_text)
    print("Number of Messages:", len(dataset))
    dataset[:3]
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok, everyting looks good. As expected we have 5,574 messages. We can access each message and it's classification individually by indexing the list.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split the Data
    Next, we'll split the dataset into two datasets. We'll take 80% of the messages for our training dataset and we'll reserve 20% of the messages for our test dataset.
    """)
    return


@app.cell
def _(dataset, math, random):
    # We can use type aliases to represent complext types.
    DatasetType = list[tuple[str, str]]

    def split_dataset(
        dataset: DatasetType,
        test_fraction: float = 0.2
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Split a dataset into a test and training dataset.

        Randomly shuffle the records before splitting the dataset. 

        Args:
            dataset: The dataset that will be split.
            test_fraction: The fraction of messages that will be
                placed in the test dataset.

        Returns:
            A 2-item tuple.
            1. The first item is the test dataset.
            2. The second item is the training dataset.
        """
        qty_test_messages = math.floor(len(dataset) * test_fraction)
        random.shuffle(dataset)  # Randomly shuffle the text messages in place.
        # Return the test and training datasets as a tuple.
        return dataset[:qty_test_messages], dataset[qty_test_messages:]

    test_data, train_data = split_dataset(dataset)
    return test_data, train_data


@app.cell
def _(test_data, train_data):
    print("Number of test messages:", len(test_data))
    print("Number of training messages:", len(train_data))
    return


@app.cell
def _(train_data):
    n_ham = sum(1 for msg in train_data if msg[0] == "ham")
    n_spam = sum(1 for msg in train_data if msg[0] == "spam")
    n_ham, n_spam
    return n_ham, n_spam


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok, that looks about right.  1114 + 4460 add up to 5,574. We'll set the test data aside and work with the training data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training a Naive Bayes Model

    NB models are classification models. That means they take an input (the text message) and assign the message to one of several classes. In this exercise, our two classes are *spam* and *ham*.

    /// admonition
    There are other types of machine learning models. Some are regression models, which means they return a number, and others are generative, which means they generate text, an image, or some other artifact. LLMs are generative models.
    ///

    An NB spam classifier assumes that some words occur more frequently in spam messages than in ham messages, and vise versa. We train the model by giving it a training dataset so it can calculate word frequencies for both the spam and ham classes.

    Later, after the NB model is trained, we can give it a text message. It will extract the words from the text messages, look up the words in it's frequency tables, and calculate both a ham score and a spam score from the word frequencies. The model will assign the message to the class with the higher score.

    The final section of this notebook explains the theory behind naive Bayes models. That section is optional.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Counting Words
    We're going to create two dictionaries. For both dictionaries the key is a word that appears in a text message and the value is the number of counts.
    """)
    return


@app.cell
def _(dataset, nltk):
    # Tokenizers split text into chunks. Tokens are usually words, parts of words, numbers, or punctuation.
    # NLTK's TweetTokenizer words well for text messages.
    tokenizer = nltk.tokenize.TweetTokenizer()

    # Here's another type alias
    WordCountType = dict[str, int]

    def get_word_counts(dataset: list[tuple[str, str]]) -> tuple[WordCountType, WordCountType]:
        """Create two word count dictionaries, one for spam and one for ham.

        For both dictionaries, the keys are every word or punctuation mark that appears
        in the corresponding class's text messages. The values are the number of text
        messages that the word appears in. A word is only counted once per text message
        even if it appears multiple times in one message.

        Args:
            dataset: A list of class-message tuples.

        Returns:
            A tuple of two dictionaries.
            1. The ham dictionary.
            2. The spam dictionary.
        """
        ham_counts: WordCountType = {}
        spam_counts: WordCountType = {}
        for msg in dataset:
            counts = ham_counts if msg[0] == "ham" else spam_counts
            tokens = tokenizer.tokenize(msg[1])
            for token in set(tokens):
                # Using set means a token is only counted once, even it it occurs multiple
                # times in a message.
                if token in counts:
                    counts[token] += 1
                else:
                    counts[token] = 1
        return (
            dict(sorted(ham_counts.items(), reverse=True, key=lambda x: x[1])),
            dict(sorted(spam_counts.items(), reverse=True, key=lambda x: x[1]))
        )

    ham_counts, spam_counts = get_word_counts(dataset)
    print(ham_counts)
    return ham_counts, spam_counts, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We're also going to need a collection of all the words that appear in either a ham or spam message. We'll call this our *vocabulary*.
    """)
    return


@app.cell
def _(ham_counts, spam_counts):
    vocab = set(ham_counts.keys()) | set(spam_counts.keys())
    print("Number of words in vocabulary:", len(vocab))
    print(vocab)
    return (vocab,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have completed the training of our NB model. We can move on to inference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference
    Now that our model is trained, we can test our model against the test dataset. Remember, we split the datasets before we trained the model, so the model has not seen any of the messages in the test dataset.

    /// attention | IMPORTANT!
    Never test your model on training data. Always hold back part of your dataset for testing.
    ///

    In this section, we'll write the functions that take a text message and use the NB model to infer (that is, predict) whether the message is ham or spam. This section includes some mathematical and statistical notation that's commonly used in machine learning papers. Don't worry too much if you don't understand the notation -- just try to understand the high-level concepts. I'm including the math so you get used to seeing it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Word Probabilities
    During inference we need to estimate a word's probability of occurring for each class.

    Here is some relevant math and statistics notation:

    $$ \begin{align}
        N_{\textrm{ham}} &\coloneq \textrm{Number of ham messages in training data} \\
        N_{\textrm{spam}} &\coloneq \textrm{Number of spam messages in training data} \\
        w &\coloneq \textrm{A variable representing a word that could appear in a text message.} \\
        n_w^\textrm{ham} &\coloneq \textrm{Number of ham messages that contain word } w \\
        n_w^\textrm{spam} &\coloneq \textrm{Number of spam messages that contain word } w \\
        P(w \;|\;\textrm{ham}) &\coloneq \textrm{Probability that a ham message contains word } w \\
        P(!w \;|\;\textrm{ham}) &\coloneq \textrm{Probability that a ham message does NOT contain word } w \\
        P(w \;|\;\textrm{spam}) &\coloneq \textrm{Probability that a spam message contains word } w \\
        P(!w \;|\;\textrm{spam}) &\coloneq \textrm{Probability that a spam message does NOT contain word } w \\
    \end{align} $$

    Here are the formulas for estimating word probabilities for each class.

    $$ \begin{align}
        P(w \;|\;\textrm{ham}) &= \frac{n_w^\textrm{ham} + 1}{N_{\textrm{ham}} + 2} \\
        P(w \;|\;\textrm{spam}) &= \frac{n_w^\textrm{spam} + 1}{N_{\textrm{spam}} + 2} \\
        P(!w \;|\;\textrm{ham}) &= 1 - P(w \;|\;\textrm{ham}) \\
        P(!w \;|\;\textrm{spam}) &= 1 - P(w \;|\;\textrm{spam})
    \end{align} $$

    You might have been thinking that $P(w \;|\;\textrm{spam})$ should equal $n_w^\textrm{ham} / N_{\textrm{spam}}$. Adding 1 to the numerator and 2 to the denominator is called *Laplace smoothing*. See the final section of this notebook for an explanation of why Laplace smoothing is necessary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculate a Word Probability
    The `calculate_probability` function estimates the probability that a word appears (or if `appears=False`, then doesn't appear) in a spam or ham message.

    Note line 5.
    * To get a count from the dictionary, we use the syntax `counts.get(word, 0)`. The `get` method will return 0 if the word isn't in the dictionary, which is what we want.
    * Also, you can see that we're doing Laplace smoothin.
    """)
    return


@app.cell
def _(ham_counts, n_ham, n_spam, spam_counts):
    def calculate_probability(
        word: str,
        for_spam: bool = True,
        appears: bool = True
    ) -> tuple[float, float]:
        """Calculate probability of a word appearing or not appearing in a message.

        Args:
            word: The word for which the proababilities are calculated.
            for_spam: Calculate spam probabilities if True (default), otherwise
                calculate ham probabilities.
            appears: Calculate the probability of the word appearing in a message
                if True (default), or NOT appearing in the message if False.

        Returns:
            The probability as a floating point value.
        """
        counts = spam_counts if for_spam else ham_counts
        qty_messages = n_spam if for_spam else n_ham
        appears_prob = (counts.get(word, 0) + 1) / (qty_messages + 2)
        if appears:
            return appears_prob
        else:
            return 1 - appears_prob  # Probability word does not appear

    return (calculate_probability,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's test the function on a common word. Here are all the probabilities for the word *to*.
    """)
    return


@app.cell
def _(calculate_probability):
    print("Probability that 'to' appears in spam:", calculate_probability("to", for_spam=True))
    print("Probability that 'to' doesn't appear in spam:", calculate_probability("to", for_spam=True, appears=False))
    print("Probability that 'to' appears in ham:", calculate_probability("to", for_spam=False))
    print("Probability that 'to' doesn't appear in ham:", calculate_probability("to", for_spam=False, appears=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculate the Class Score
    The `get_class_score` function calculates a numeric ham or spam score for a text message.

    Calculating the score requires us to calculate the prior probability that a message is spam or ham. By prior probability, we mean the probability BEFORE we evaluate the words in the message. For example, if we know that on average, 13% of text messages are spam, then before we look at the message the probability that it is spam is 13%. Since the prior probabilities are the same for every word, we calculate them once, outside the function definition.

    Also, we're going to use the logarithms of the probabilities to calcuate our score. The reason is explained in the final section.

    Here's how the score is calculated:
    1. Set the score equal to the natural log of the class probability.
    2. For every word in the training vocabulary, check if the word is in the message.
    3. If the word is in the message, add the natural log of the probability that the word is in the message to the score.
    4. If the word is NOT in the message, add the natural log of the probability that the word is NOT in the message to the score.
    5. That's it. Return the score.
    """)
    return


@app.cell
def _(calculate_probability, math, n_ham, n_spam, train_data, vocab):
    # Probability that a message is spam or ham if we ignore it's content
    #   This is called a prior probability.
    prob_spam = n_spam / len(train_data)
    prob_ham = n_ham / len(train_data)

    # For reasons that will be explained in the last section, our class
    #   score will be based on the logarithm of the probabilities.
    log_prob_spam = math.log2(prob_spam)
    log_prob_ham = math.log2(prob_ham)


    def get_class_score(message_words: list[str], for_spam: bool = True) -> float:
        """Get the ham or spam score for a list of words.

        Args:
            message_words: A list of words that appear in a message. The list should
                not have any duplicates.
            for_spam: Calculate the score for the spam class if True (default), or
            for ham otherwise.

        Returns:
            The class score. All scores are negative. Higher numbers (closer to zero)
            correspond to higher probabilities.
        """
        score = log_prob_spam if for_spam else log_prob_ham
        for word in vocab:
            score += math.log2(
                calculate_probability(
                    word,
                    for_spam=for_spam,
                    appears=word in message_words
                )
            )
        return score

    return get_class_score, prob_ham, prob_spam


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Predict the Class
    Finally we are ready to predict whether a message is spam. The `is_spam` function splits a message into words and calculates a ham score and a spam score. If the spam score is higher than the ham score, the model predits spam.
    """)
    return


@app.cell
def _(get_class_score, tokenizer):
    def is_spam(message: str, verbose: bool = False) -> bool:
        """Predict whether a text message is spam.

        Args:
            message: A text message.
            verbose: Display the ham and spam scores if True.
                Optional, default is False.

        Returns:
            True if model predicts message is spam. False otherwise.
        """
        message_words = set(tokenizer.tokenize(message))
        spam_score = get_class_score(message_words, True)
        ham_score = get_class_score(message_words, False)
        if verbose:
            print("Ham:", ham_score, "Spam:", spam_score)
        return spam_score > ham_score

    return (is_spam,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's test the `is_spam` function on a message that probably isn't spam.
    """)
    return


@app.cell
def _(is_spam):
    is_spam("Do you want to get something to eat?", verbose=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why are the scores negative? Probabilites range from 0 to 1.0. A logarithm of any number less than 1.0 is negative, so our scores are negative. Note that the score for ham is still greater than the score for spam.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation

    It's time to see how our NB model does on the training data. We'll evalute the model on the *test* dataset. Remember, the test dataset was not used when we trained the NB model. To quantify how well the model performs, we're going to predict the class of every message in the dataset and count the number of correct and incorrect predictions for each class.

    ### Terminology
    #### Positive vs Negative
    If the NB model predicts that a message is spam, we call that a *positive* result. A ham prediction is a *negative* result. This is counterintuitive because receiving spam doesn't put most of us in a positive mood. This terminology the same as what's used in medical screening tests. For example, suppose that we take a genetic test for photic sneeze reflex. A *positive* result means the test indicates we have a genetic variation that causes sneezing when exposed to bright light, and a *negative* result indicates we don't have the condition.

    #### True vs False
    If a prediction is correct, we say it's *true*. If incorrect, we say it's *false.

    #### Putting it All together
    | Category | Description |
    | ---------| ------------|
    | True Positive | Model predicted spam and the message is spam |
    | True Negative | Model predicted ham and the message is ham |
    | False Positive | Model predicted spam and the message is ham |
    | False Negative | Model predicted ham and the message is spam |

    ### Results Class
    The next cell defines the `Results` dataclass, which will help us track model results.  We'll discuss the meaning of *precision*, *recall*, *accuracy*, and *f1* in the section on evaluation metrics.
    """)
    return


@app.cell
def _(dataclasses):
    @dataclasses.dataclass
    class Results:
        """Evaluation results for a model."""
        true_positives: int
        """Number of spam messages that the model predicted are spam."""
        true_negatives: int
        """Number of ham messages that the model predicted are ham."""
        false_positives: int
        """Number of ham messages that were predicted to be spam."""
        false_negatives: int
        """Number of spam messages that were predicted to be ham."""

        false_positive_messages: list[str]
        """Messages that were incorrectly predicted to be spam."""
        false_negative_messages: list[str]
        """Messages that were incorrectly predicted to be ham."""


        @property
        def total(self) -> int:
            """Total number of predictions."""
            return (
                self.true_positives + self.true_negatives +
                self.false_positives + self.false_negatives
            )

        @property
        def accuracy(self) -> float:
            """Accuracy metric."""
            return (self.true_positives + self.true_negatives) / (self.total)
    
        @property
        def precision(self) -> float:
            """Precision metric."""
            return self.true_positives / (self.true_positives + self.false_positives)

        @property
        def recall(self) -> float:
            """Recall metric."""
            return self.true_positives / (self.true_positives + self.false_negatives)

        @property
        def f1(self) -> float:
            """F1 metric."""
            return 2 / ((1/self.precision) + (1/self.recall))
    

    return (Results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation Loop
    The `evaluate_model` loops over the test dataset, makes predictions, and returns the evaluation results. We'll also store messages with incorrect (false) predictions.
    """)
    return


@app.cell
def _(Results, is_spam, mo, test_data):
    def evaluate_model() -> Results:
        """Evaluate the NB model.

        Returns:
            A Results object.
        """
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        false_positive_messages = []
        false_negative_messages = []

        for label, message in mo.status.progress_bar(
            test_data,
            title="Evaluating!",
            subtitle="Please wait a moment ..."
        ):
            spam_pred = is_spam(message)
            match (label, spam_pred):
                case ("spam", True):
                    # Message is spam and model predicted spam (Correcte!)
                    true_positives += 1
                case ("ham", True):
                    # Message is ham but model predicted spam (Wrong!)
                    false_positives += 1
                    false_positive_messages.append(message)
                case ("spam", False):
                    # Message is spam but model predicted ham (Wrong!)
                    false_negatives += 1
                    false_negative_messages.append(message)
                case ("ham", False):
                    # Message is ham and model predicted ham (Correc!)
                    true_negatives += 1

        return Results(
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            false_positive_messages=false_positive_messages,
            false_negative_messages=false_negative_messages
        )

    results = evaluate_model()
    print(results)
    return (results,)


@app.cell(hide_code=True)
def _(mo, results):
    mo.md(rf"""
    ### Results
    Since we randomly selected the messages for the test and training datasets, these results will vary slightly every time we run the notebook. Every time I ran the notebook, the model correctly classifed all but about twenty of the 1,114 text message in the test dataset.

    Let's look at the results in terms of percentages.

    | Actual Message Class | Predicted to be Spam | Predicted to be Ham |
    | ---------------------| -------------------- | ------------------- |
    | Spam | {results.true_positives / results.total:.3f} | {results.false_negatives / results.total:.3f} |
    | Ham  | {results.false_positives / results.total:.3f} | {results.true_negatives / results.total:.3f} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation Metrics
    This section uses some abbreviations.
    * TP: Number of True Positives
    * TN: Number of True Negatives
    * FP: Number of False Positives
    * FN: Number of False Negatives

    #### Accuracy
    *Accuracy* is the proportion of predictions that are correct.
    $$ \textrm{accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

    Our predictions are correct {(results.true_positives + results.true_negatives) / results.total:.2%} of the time! That's an impressive result for such a simple dataset.

    #### Recall
    *Recall* is the proportion of positives in the dataset that our model correctly predicts to be positive. In this example, it's the number of correctly-predicted spam messages divided by the total number of spam messages in the datase.

    $$ \textrm{recall} = \frac{TP}{TP + FN} $$

    #### Precision
    *Precision* is the proportion of positive predictions that actually are positive. In this example precision is the number of correctly-predicted spam messages divided by the total number of spam predictions.

    $$ \textrm{precision} = \frac{TP}{TP + FP} $$

    #### Precision and Recall Example
    Suppose we trained a machine learning model to evaluate images and predict whether an image protrayed a dog. The following graphic shows how we would calculate precision and recall. (Wikipedia. *Precision and Recall*. https://en.wikipedia.org/wiki/Precision_and_recall, Accessed on 19 April 2026. )

    <img src="public/precision-recall.png", width="400" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### F1 Score
    The precision and recall metrics can be combined into an F1 score. The F1 score is the harmonic mean of precision and recall. (Wikipedia. *F-Score*, https://en.wikipedia.org/wiki/F-score, Accessed on 19 April)


    $$ F_1 = \frac{2}{\frac{1}{\textrm{recall}} + \frac{1}{\textrm{precision}}} = \frac{2TP}{2TP + FP + FN} $$
    """)
    return


@app.cell(hide_code=True)
def _(mo, results):
    mo.md(rf"""
    ### Naive Bayes Results
    | Metric | Value |
    |--------| ------|
    | Accuracy | {results.accuracy:.4f} |
    | Precision | {results.precision:.4f} |
    | Recall | {results.recall:.4f} |
    | F1 | {results.f1:.4f} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Machine Learning Concepts
    A machine learning algorithm is an algorithm that learns how to complete a task by examining data. In this exercise our naive Bayes spam filter examined a dataset containing both spam and non-spam messages and chose parameters based on the datatase.

    #### Classification, Regression, and Generation
    The naive Bayes model in this exercise is a *classification* model. It assigns an input (text message) to a class (ham or spam). There are many types of classification models. They can use many different types of input, including text, numbers, audio files, and imagery.

    Many models are *regression* models. Regression models generate a numeric value. A machine learning model that attempts to predict how many district ranking a points an FRC team will earn would be a regression model.

    Large language models, like ChatGPT and CoPilot are generative models. They generate text and imagery from text and image inputs.

    ### Supervised, Unsupervised, and Self-Supervised Learning
    Specifically, the naive Bayes algorithm is a *supervised* machine learning algorithm, which means it needs to be trained on a pre-labled dataset (for classification) or answers. There are algorthms that are *unsupervised*, which means they don't need a labled dataset. Large langague models us a *self-supervised* learning process, which means they are able to generate their own training answers.

    ### Training and Inference
    The process of examing the training data so the model can learn to complete a task is called *training.* In this exercise, training consisted of scanning the training dataset and counting word appearances in spam and ham messages.

    *Inference* happens when we ask a trained model to make a prediction.

    ### Parameters
    Parameters are numbers that are part of a machine learning model that are adjusted during the training process. In this exercise the parameters are the word counts.

    #### Generalize
    *Generalize* refers to a model's performance on various kinds of input. A model that performs well on training data but performs poorly on inputs it hasn't seen before generalizes poorly.

    | Concept | Description |
    | ------: | :---------- |
    | Classification Model | Assigns an input to a class or category |
    | Regression Model | Predicts a numeric value |
    | Generative Model | Generates text, imagry, or other artifacts |
    | Supervised Model | Requires labeled datasets for training |
    | Unsupervised Model | Doesn't need any labels to learn from data |
    | Self-Supervised Model | Generates its own labels or answers to use during training |
    | Training | The act of learning to complete a task by evaluating data |
    | Inference | Make predictions from from inputs (occurs after training is complete |
    | Parameters | Numeric values within a model that are set and revised during training |
    | Generalize | Make inferences from a wide range of inputs, including inputs that are different from training data |
    | Accuracy | The proportion of predictions that are correct |
    | Precision | The proportion of positive predictiosn that are correct |
    | Recall | The proportion of actual positives that are predicted to be positive |
    | F1 | The harmonic mean of recall and precision |
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Some Statistics Terminolgy

    Naive Bayes is a classification model, which means it assigns messages to different categories, or *classes*. In this example we have two classes, spam and ham. Per equation (1), we're using the random variable $C$ to represent the class of a message.

    $$ C \coloneq \textrm{message class, either "ham" or "spam"} \tag{1} $$

    FYI, the symbol $\coloneq$ indicates a definition. In the first statement we're defining the random variable variable $C$

    /// admonition | Advanced Concept
    In statistics, random variables are represented with capitol letters. The whole point of a random variable is that we don't know it's actual value -- it's like the value of a die roll *before* we roll the die. So $C$ represents the class of a message before we know whether it's spam or ham. We use lowercase variables to represent known values. So we could use the variable $c$ to represent the class of a message when we know what the class is.
    ///

    The syntax $P(X = x)$ in statistics means *the probability that random variable $X$ will assume the value $x$*. So $P(C = \textrm{spam})$ means *the probability that the message is spam.*

    $$ \begin{align}
        P(C = \textrm{spam}) &\coloneq \textrm{Probability that a message is spam} \tag{2} \\
        P(C = \textrm{ham}) &\coloneq \textrm{Probability that a message is not spam} \tag{3}
    \end{align} $$

    Since we already counted the number of spam and ham messages in our dataset, we can easily calculate the probabilities that a message is spam or ham.
    """)
    return


@app.cell(hide_code=True)
def _(mo, prob_ham, prob_spam):
    mo.md(rf"""
    $$ P(c = \textrm{{ham}}) = {prob_ham:.4f} \tag{4} $$
    $$ P(c = \textrm{{spam}}) = {prob_spam:.4f} \tag {5} $$


    So if we receive a text message and we don't bother to read it, we would estimate that there is an 86% probability that the message is ham, and a 14% probability that it's spam. In Bayesian statistics, we would call these *prior probabilities*.

    Another way to look at this is that if we made a model that always guessed a message is ham, it would be right 86% of the time. That's a passing score on most exams, but obviously we need to try to do better.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you may have guessed from our word counting activities, we're going to improve our classification model by considering the words that appear in each message. So it's time for more math symbols.

    $$ \begin{align}
        W & \coloneq \textrm{Random variable representing a word that could appear in a message} \tag{6} \\
        w & \coloneq \textrm{A specific word that is known to be in a message} \\
        i & \coloneq \textrm{Subscript showing the position of a word in a message} \tag{7} \\
        w_i & \coloneq \textrm{The ith word in a message.} \tag{8} \\
        P(W_i = w) &\coloneq \textrm{Probability that the ith word in a message is some specific word } w
    \end{align} $$

    So for a message like "Don't forget to buy bread"

    $$ \begin{align}
        w_0 &= \textrm{Don't}  \tag{9} \\
        w_1 &= \textrm{forget} \tag{10} \\
        w_2 &= \textrm{to} \tag{11} \\
        w_3 &= \textrm{buy} \tag{12} \\
        w_4 &= \textrm{bread} \tag{13} \\
    \end{align} $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimating Word Probabilities
    Knowing the exact probability of a word appearing in a text message would require that we have a dataset containing every text message that has been sent, and that will be sent in the future. Obviously that's impossible. But we can estimate probabilities using our training dataset.

    $$ \begin{align}
        P(W = w) &\coloneq \textrm{Probability that word } w \textrm{ appears in a message.} \\
        n_w &\coloneq \textrm{Number of messages in which word } w \textrm{ appears} \\
        N &\coloneq \textrm{Total number of messages in training dataset (both spam and ham)} \\
        P(W = \textrm{w}) &\approx n_w / N
    \end{align} $$

    Let's estimate the probability of the word "Don't appearing in a text message."
    """)
    return


@app.cell
def _(dataset, ham_counts, spam_counts):
    N = len(dataset)

    def show_word_probability(word):
        word_count = ham_counts.get(word, 0) + spam_counts.get(word, 0)
        print(f"Occurrences of the word {word}:")
        print("\tHam message count:", ham_counts.get(word, 0))
        print("\tSpam message count:", spam_counts.get(word, 0))
        print("\tTotal Count:", word_count)
        print("\nTotal Messages:", N)
        word_prob = word_count / N
        print(f"P(w = '{word}') = {word_prob:.5f}")

    show_word_probability("Don't")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conditional Probabilities
    The main premise of a Naive Bayes spam classifier is that some words are more likely to occur in spam messages than in ham messages, and vice-versa. So it's nice to know that the probability of the word *Don't* occurring in a text message is about 0.5%, but what we really want to know know is what is the probability of it occurring in spam or ham messages. We use conditional probabilities to represnt this mathematically.

    $$ \begin{align}
        P(W = \textrm{Don't} \; | \; C = \textrm{ham}) \coloneq \textrm{Probability of "Don't" occurring in a ham message} \\
        P(W = \textrm{Don't} \; | \; C = \textrm{spam}) \coloneq \textrm{Probability of "Don't" occurring in a spam message}
    \end{align} $$

    $P(X = x)$ still refers to the probability that event $x$ will occur. The pipe, or vertical bar symbol ($|$) can be read as "given that". In statistics, when you see the syntax $P(X = x \; | \; Y = y)$ it means the probability of event $x$ occurring given that $y$ has already occurred.
    """)
    return


@app.cell
def _(ham_counts, n_ham, n_spam, spam_counts):
    def calc_ham_prob(word):
        count = ham_counts.get(word, 0)  # Count is zero if word not in dictionary
        return count / n_ham

    def calc_spam_prob(word):
        count = spam_counts.get(word, 0)  # Count is zero if word not in dictionary
        return count / n_spam

    _word = "Don't"
    calc_ham_prob(_word), calc_spam_prob(_word)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flip the Probs
    It's easy to estimate the probability of a word occurring in a message given that it's spam or ham. But what we really need is the probability that a message is spam or ham given that it contains one or more specific words. We have $P(W = W \;|\; C = c)$, but we need $P(C = c \;|\; W = w)$. Fortunately there is a simple formula to calculate $P(C = c \;|\; W = w)$ given $P(W = w \;|\; C = c)$. It's called *Bayes Theorem.*

    $$ P(C = c \;|\; W = w) = \frac{P(W = w \;|\; C = c) \cdot P(C = c)}{P(W = w)} $$

    ## But Messages Have More Than One Word
    The probability that a spam message contains a word lke "Hey" is just the number of spam messages that contain "Hey" divided by the total number of spam messages. That's easy. But what's the probability that a message contains both the words "Hey" and "there"?

    Here's where we make two big simplifying assumptions.
    1. We assume that the order of the words in the text doesn't matter. Models that use this assumption are called *bag-of-words* models.
    2. We're going to assume that the probability of any word appearing in a message is completely independent of all other words in the message.

    These assumptions are obviously not true. We know that changing the order of words can drastically change the meaning of a sentence, and I suggest that any message containing the word "apple" is more likely to contain the word "pie" than most other messages. The second assumption is why the algorithm is called *Naive.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Abbreviated Syntax
    We're going to abbreviate the probability syntax.
    $$P(c \;|\; w) \coloneq P(C = c \;|\; W = w) $$
    ///

    Anyway, with these two assumptions in place, then the probability of a set of $m$ words appearing in a message is:

    $$ P(c \;|\; (w_0, w_1, ..., w_m)) = \frac{P(c)\prod_j^{|V|} P(x_j = 1 | c)^{x_j} P(x_j = 0 | c)^{1-x_j}}{P(w_0, w_1, ..., w_m)}$$

    $|V|$ is the number of words in our vocabulary. The vocabulary $V$ is the set of words that appeared in the training data set.

    $x_j$ is an indicator variable. $x_j = 1$ if $w_j$ is in the message and $x_j = 0$ if $w_j$ is not in the message.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$ \begin{align}
        N_{ham} &\coloneq \textrm{Number of ham messages in training data} \\
        N_{spam} &\coloneq \textrm{Number of spam messages in training data} \\
        P(\textrm{ham}) &\coloneq \textrm{Probability that a message is ham} \\
        P(\textrm{spam}) &\coloneq \textrm{Probability that a message is sam} \\
        P(\textrm{ham}) &= N_{ham}
    \end{align} $$
    """)
    return


if __name__ == "__main__":
    app.run()
