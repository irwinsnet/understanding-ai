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

    return (get_class_score,)


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

    ### Classification, Regression, and Generation
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

    ### Hyperparameters
    ...

    ### Generalize
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Naive Bayes Theory (Optional)
    The rest of this notebook discusses the theory behind the NB algorithm -- how and why it works. This material isn't required for understanding follow-on topics, so you can skim it or skip it if you like.

    ### Word Probabilities
    The NB algorithm estimates probabilities that specific words will appear in ham or spam messages during training. Then during inference it uses the word probabilities to estimate the likelihood that a message is ham or spam.

    Before we proceed further, let's define our notation for talking about word probabilities.

    $$ \begin{align}
        w &\coloneq \textrm{A variable that represents a single word} \\
        n_w &\coloneq \textrm{The number of text messages that contain word } w \\
        M &\coloneq \textrm{The set of unique words in a message} \\
        P(w \in M) &\coloneq \textrm{Probability that the word } w \textrm{ appears in message } M \\
        P(w \notin M) &\coloneq \textrm{Probability that the word } w \textrm{ does NOT appear in message } M \\
        P(w \in M) &=  \frac{n_w}{N} \\
        P(w \notin M) &= 1 - P(w \in M)
    \end{align} $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conditional Probabilities
    The word probabilities in the previous section are *unconditional* probabilities. That is, they ignore the condition of whether a message is ham or spam. A close inspection of the NB training code reveals that it doesn't calculate any unconditional probabilities. The NB training code calculates separate word probabilities for ham and spam messages. Probabilities that depend on some condition (like whether a message is ham or spam) are called *conditional* probabilities.

    Here is our notation for conditional word probabilities.

    $$ \begin{align}
        n_w^\textrm{ham} &\coloneq \textrm{The number of ham messages that contain the word } w \\
        n_w^\textrm{spam} &\coloneq \textrm{The number of spam messages that contain the word } w \\
        P(w \in M \;|\; \textrm{ham}) &\coloneq \textrm{Probability that message } M \textrm{ contains word } w \textrm{ given that message } M \textrm{ is a ham message} \\
        P(w \in M \;|\; \textrm{spam}) &\coloneq \textrm{Probability that message } M \textrm{ contains word } w \textrm{ given that message } M \textrm{ is a spam message} \\
        P(w \notin M \;|\; \textrm{ham}) &\coloneq \textrm{Probability that message } M \textrm{ does not contain word } w \textrm{ given that message } M \textrm{ is a ham message} \\
        P(w \notin M \;|\; \textrm{spam}) &\coloneq \textrm{Probability that message } M \textrm{ does not contains word } w \textrm{ given that message } M \textrm{ is a spam message} \\
        P(w \in M \;|\; \textrm{ham}) &= n_w^\textrm{ham} / N_\textrm{ham} \\
        P(w \in M \;|\; \textrm{spam}) &= n_w^\textrm{ham} / N_\textrm{spam} \\
        P(w \notin M \;|\; \textrm{ham}) &= 1 - P(w \in M \;|\; \textrm{ham}) \\
        P(w \notin M \;|\; \textrm{spam}) &= 1 - P(w \in M \;|\; \textrm{spam})
    \end{align} $$

    The vertical bar character ('|') indicates that a probability is a conditional probability. It's often called a *pipe*. The expression on the left side of the pipe is the item for which we are calculating the probability, and the expression on the right is the condition that the probability depends on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Probabilities for Multiple Words at Once
    Probabilities for individual words appearing in text messages are useful, but most text messages consist of more than one word. What is the probability that the words "Don't", "forget", "to", "buy", "bread", "." and NO other words appear in a text message?

    To figure that out, we're going to make two simplifing assumptions.
    1. The order of words doesn't matter. So the NB algorithm considers "Don't forget to buy bread" and "Don't buy bread to forget" to be identical text messages. Models that ignore word order are called *Bag of Words* models.
    2. The probability of a word appearing in a message is independent of the the other words in the message. In reality this isn't true. It's likely that text messages that contain the word "peanut" are more likely to contain the word "butter" or "sandwich" than other messages. In fact, this assumption is why the algorithm is called NAIVE Bayes.

    Regardless, the algorithm works surprisingly well in spite of these simplifications.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is some more notation that we'll need to calculate probabilities of groups of words appearing in a message.

    $$ \begin{align}
        V &\coloneq \textrm{The model's vocabulary, i.e., the number of words in the training data} \\
        |V| &\coloneq \textrm{The number of words in the model's vocabulary} \\
        i &\coloneq \textrm{An index variable that ranges from 1 to } |V| \\
        M &\coloneq \textrm{The set of unique words in a message} \\
        |M| &\coloneq \textrm{The number of unique words in a message} \\
        j &\coloneq \textrm{An index variable that ranges from 1 to } |M| \\
        w_j &\coloneq \textrm{A variable that represents the jth unique word in a message.} \\
        P(\{w_1, w_2, ..., w_j\} \in M) &\coloneq \textrm{Probability that the words } \{w_1, w_2, ..., w_j\} \textrm{ and NO OTHER WORDS appear in message } M \\
    \end{align} $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok, here we go.

    $$ \begin{align}
    x_i \coloneq \textrm{Indicator variable that equals 1 if } w_i \textrm{ is in message } M \textrm{ and 0 otherwise} \\
    P(\{w_1, w_2, ..., w_i\} \in M \;|\; \textrm{ham/spam}) = \Pi_i^{|V|} P(w_i \in M \;|\; \textrm{ham/spam})^{x_i}P(w_j \notin M \;|\; \textrm{ham/spam})^{1 - x_i}
    \end{align} $$

    The capitol Pi letter, $\Pi$, indicates a product. The preceding equation says that for every unique word in the text message, we should calculate the probability of that word appearing in a message and then multiply all those probabilities together. Then for ever word in the training data (the vocabulary) that did NOT appear in the message, we should multiply the product with the probabilities of those words NOT appearing in the message. We end up with the probability of that exact set of words appearing in a message.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Flipping the Probability with Bayes Theorem
    #### What's Bayes Theorem?
    We can calculate $P(\{w_1, w_2, ..., w_i\} \in M \;|\; \textrm{spam})$, but what we really want is this:

    $$ P( \textrm{spam} \;|\; \{w_1, w_2, ..., w_i\} \in M)$$

    We have the probability that a message contains specific words given that it's spam, but what we want is the probability that a message is spam given that it contains specific words. Notationally, we just swap the expressions on the left and right side of the pipe.

    *Bayes Theorem* allows us to express $P(a \;|\; b)$ in terms of $P(b \;|\; a)$.

    $$ P(a \;|\; b) = \frac{P(b \;|\; a)P(a)}{P(b)} $$

    Here's a numeric example from the [statology.org website](https://www.statology.org/bayes-theorem-explained-simply/). You are a doctor trying to determine if someone has the flu given that they have a cough. (The following numbers are all made up.)

    * $P(A)$ is the probability that someone in the general population has the flu. This is called the *prior probability*. $P(A) = 0.1$
    * The likelihood of having a cough if you have the flu, $P(B \;|\; A)$ is 0.8.
    * The total probability of having a cough (for any reason, not just the flu) is $P(B) = 0.3$

    $$ P(a \;|\; b) = \frac{P(b \;|\; a)P(a)}{P(b)} = \frac{0.8 \times 0.1}{0.3} = 0.267$$

    In general, $P(a \;|\; b) \neq P(b \;|\; a)$. That's evident in this example, where $P(a \;|\; b) = 0.267$ and $P(b \;|\; a) = 0.8$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Applying Bayes Theorem to Spam Probabilities

    Here's what we want (using the notation that was introduced earlier in this notebook):

    $$ \begin{align}
    P(\textrm{spam} \;|\; \{w_1, w_2, ..., w_j\} \in M) &\coloneq \textrm{Probability that message } M \textrm{ is spam given that it contains specific words } \\
    P(\textrm{ham} \;|\; \{w_1, w_2, ..., w_j\} \in M) &\coloneq \textrm{Probability that message } M \textrm{ is ham given that it contains specific words} \\
    \end{align} $$

    To predict whether a message $M$ is spam or ham, we calculate both probabilities and predict the class with the highest probability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we plug our probabilities into Bayes theorem, we get the following:

    $$ \begin{align}
        P(\textrm{spam} \;|\; \{w_1, w_2, ..., w_j\} \in M) &= \frac{P(\{w_1, w_2, ..., w_j\} \in M \;|\; \textrm{spam}) \cdot P(\textrm{spam})}{P(\{w_1, w_2, ..., w_j\})} \\
        P(\textrm{ham} \;|\; \{w_1, w_2, ..., w_j\} \in M) &= \frac{P(\{w_1, w_2, ..., w_j\} \in M \;|\; \textrm{ham}) \cdot P(\textrm{ham})}{P(\{w_1, w_2, ..., w_j\})}
    \end{align}$$

    Look at the two preceding equations. Can you see a way to simplify the math?
    * We don't care about the exact probabilities that these two equations generate. We just want to pick the class (spam or ham) that has the higher probability.
    * Both equations have the exact same denominator.
    * The denominator is a probability, so it is a postive value ranging from 0 to 1.

    Dropping the denominator from both equations won't change which class has the higher value. But we're calculating a class score instead of an actual probability, so we'll use $S()$ instead of $()$.

    $$ \begin{align}
        S(\textrm{spam} \;|\; \{w_1, w_2, ..., w_j\} \in M) &= P(\{w_1, w_2, ..., w_j\} \in M \;|\; \textrm{spam}) \cdot P(\textrm{spam}) \\
        S(\textrm{ham} \;|\; \{w_1, w_2, ..., w_j\} \in M) &= P(\{w_1, w_2, ..., w_j\} \in M \;|\; \textrm{ham}) \cdot P(\textrm{ham})
    \end{align}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Definition
    In Bayesian statistics, $P(\textrm{spam})$ and $P(\textrm{ham})$ are called *prior* probabilities. They are the probabilities of a message being ham or spam that we would use **prior** to looking at words in the message.
    ///
    Calculating $P(\textrm{spam})$ and $P(\textrm{ham})$ is easy. It's just the number of spam or ham messages divided by the total number of messages in the training dataset, which we aready did in section ??.

    We calculated $P(\{w_1, w_2, ..., w_i\} \in M \;|\; \textrm{ham/spam})$ in section ??. So putting it all together, we get:


    $$ \begin{align}
        S(\textrm{spam} \;|\; \{w_1, w_2, ..., w_j\} \in M) &=  \prod_j^{|V|} P(x_j = 1 \, | \, \textrm{spam})^{x_j} P(x_j = 0 | c)^{1-x_j} \cdot n_{\textrm{spam}} / N\\
        S(\textrm{ham} \;|\; \{w_1, w_2, ..., w_j\} \in M) &=  \prod_j^{|V|} P(x_j = 1 \,|\, \textrm{ham})^{x_j} P(x_j = 0 | c)^{1-x_j} \cdot n_{\textrm{ham}} / N
    \end{align} $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$ \begin{align}
        S(\textrm{spam} \;|\; \{w_1, w_2, ..., w_j\} \in M) &=  \prod_j^{|V|} \left[ \left(\frac{n_{w_j}^{\textrm{spam}}}{N_\textrm{spam}}\right)^{x_j} \cdot \left(1 - \frac{n_{w_j}^{\textrm{spam}}}{N_\textrm{spam}}\right)^{1-x_j} \cdot \frac{n_{\textrm{spam}}}{N} \right] \\
        S(\textrm{spam} \;|\; \{w_1, w_2, ..., w_j\} \in M) &=  \prod_j^{|V|} \left[ \left(\frac{n_{w_j}^{\textrm{spam}}}{N_\textrm{ham}}\right)^{x_j} \cdot \left(1 - \frac{n_{w_j}^{\textrm{ham}}}{N_\textrm{ham}}\right)^{1-x_j} \cdot \frac{n_{\textrm{ham}}}{N} \right] \\
    \end{align} $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Laplace and the Vocabulary Problem
    1. What happens if a message contains a word that does not occur in any of our spam or ham training messages?
    2. Is this likely to happen when we use our spam filter?

    **Answers**
    1. If a message doesn't occur in any spam or any ham messages, then $n_{w_j}^{\textrm{spam}}$ and/or $n_{w_j}^{\textrm{ham}}$ will be zero, and the resulting score will be zero.
    2. Yes, it will happen all the time. Many messages will contain URLs, phone numbers, brand names, a person's name, or even mistyped words that dont' exist in our training data.

    No words actually have zero probability of appearing in a text message. The problem is that we only have so much training data, so it's practically impossible to assemble a training data set that contains every word that could possibly appear. This problem is called the *Vocabulary Problem*. It's a problem for most natural language processing techniques, not just Naive Bayes. Most Naive Bayes algorithms use *Laplace Smoothing* to solve this problem.

    /// admonition | Reference
    Brenndoerfer M., *The Vocabulary Problem: Why World-Level Tokenization Breaks Down.* https://mbrenndoerfer.com/writing/vocabulary-problem-subword-tokenization-challenges. Accessed on 25 April 2026.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Laplace Smoothing
    Suppose we are trying to predict whether the following message is spam:
    > Get the best deals at Mortimer's Used Cars!

    Also suppose that all the words in the message appear in both ham and spam training messages, except for "Mortimer." How can we deal with this? We could just remove the word "Mortimer" from the message and not calculate a probability for it, but most NB algorithms use a technique called Laplace smoothing to deal with unknown words.

    Let's pretend that our training data DID actually contain one spam message and one ham message with the word "Mortimer." Then the word probabilities would be:

    $$ P(\textrm{Mortimer} \,|\, \textrm{ham/spam}) = \frac{1}{N_{\textrm{ham/spam}} + 2} $$

    So when we're predicting whether a message is spam or ham, for every word that appears in the message, we pretend that our training dataset contained two additional messages that contained that word, one ham message, and one spam message. Consequently, no words will have zero probability. This technique is called Laplace smoothing.

    So we update our equations for ham and spam scores like so:

    $$ \begin{align}
        S(\textrm{spam} \;|\; \{w_1, w_2, ..., w_j\} \in M) &=  \prod_j^{|V|} \left[ \left(\frac{n_{w_j}^{\textrm{spam}} + 1}{N_\textrm{spam} + 2} \right)^{x_j} \cdot \left(1 - \frac{n_{w_j}^{\textrm{spam}} + 1}{N_\textrm{spam} + 2}\right)^{1-x_j} \cdot \frac{n_{\textrm{spam}}}{N} \right] \\
        S(\textrm{ham} \;|\; \{w_1, w_2, ..., w_j\} \in M) &=  \prod_j^{|V|} \left[ \left(\frac{n_{w_j}^{\textrm{ham}} + 1}{N_\textrm{ham} + 2}\right)^{x_j} \cdot \left(1 - \frac{n_{w_j}^{\textrm{ham}} + 1}{N_\textrm{ham} + 2}\right)^{1-x_j} \cdot \frac{n_{\textrm{ham}}}{N} \right] \\
    \end{align} $$

    /// admonition | Reference
    Wikipedia. *Additive Smoothing.* https://en.wikipedia.org/wiki/Additive_smoothing. Accessed on 25 April 2026.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Limits of Floating Point Math
    Most words in our model's vocabulary will have very small probabilities of occurring. For a long text message, multiplying the small probabilities for each word will result in an extremely small score. In fact, the score might be too small for the computer's CPU to represent. For a 64-bit number, the smallest number that can be represented is about $2.2 \times 10^{-308}$. This phenomenon is called underflow.

    This problem is easy to fix. First we'll modify our score functions by taking the logarithm of the score.

    $$ \begin{align}
        S(\textrm{spam} \;|\; \{w_1, w_2, ..., w_j\} \in M) &=  \log \left(\prod_j^{|V|} \left[\left(\frac{n_{w_j}^{\textrm{spam}} + 1}{N_\textrm{spam} + 2} \right)^{x_j} \cdot \left(1 - \frac{n_{w_j}^{\textrm{spam}} + 1}{N_\textrm{spam} + 2}\right)^{1-x_j}  \right] \cdot \frac{n_{\textrm{spam}}}{N} \right) \\
        S(\textrm{ham} \;|\; \{w_1, w_2, ..., w_j\} \in M) &= \log \left( \prod_j^{|V|} \left[ \left(\frac{n_{w_j}^{\textrm{ham}} + 1}{N_\textrm{ham} + 2}\right)^{x_j} \cdot \left(1 - \frac{n_{w_j}^{\textrm{ham}} + 1}{N_\textrm{ham} + 2}\right)^{1-x_j}  \right] \cdot \frac{n_{\textrm{ham}}}{N}\right) \\
    \end{align} $$

    /// admonition | Reminder
    $x_j$ is an indicator variable that is 1 if word $w_j$ is in the text message and 0 if it is not.
    ///

    A convenient thing about logarithms is that the logarithm of a product of many terms is equal to the sum of the logarithms of the individual terms. So the final versions of our scoring functions becomes:

    $$ \begin{align}
        S(\textrm{spam} \;|\; \{w_1, w_2, ..., w_j\} \in M) &=  \sum_j^{|V|} \left[{x_j}\log \left(\frac{n_{w_j}^{\textrm{spam}} + 1}{N_\textrm{spam} + 2} \right) + (1-x_j) \log \left(1 - \frac{n_{w_j}^{\textrm{spam}} + 1}{N_\textrm{spam} + 2}\right)  \right]  + \log \left( \frac{n_{\textrm{spam}}}{N} \right) \\
        S(\textrm{ham} \;|\; \{w_1, w_2, ..., w_j\} \in M) &=  \sum_j^{|V|} \left[{x_j}\log \left(\frac{n_{w_j}^{\textrm{ham}} + 1}{N_\textrm{ham} + 2} \right) + (1-x_j) \log \left(1 - \frac{n_{w_j}^{\textrm{ham}} + 1}{N_\textrm{ham} + 2}\right)  \right]  + \log \left( \frac{n_{\textrm{ham}}}{N} \right)
    \end{align} $$

    These are the equations that are implemented in our Python code. The base of the logarithms doesn't matter -- you can use base 2, base 10, natural logs, whatever.

    Since
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
