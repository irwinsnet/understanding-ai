import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Understanding Artificial Intelligence
    # Part I: Terms and Concepts
    We encounter many differerent terms when reading about artificial intelligence (AI). This notebook discusses AI and large language models, but here are a few of the terms we'll come across in subsequent notebooks:

    * Machine Learning
    * Natural Language Processing
    * Neural Nets
    * Deep Learning
    * Reinforcement Learniing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Artificial Intelligence
    Humans have imagined machines that behave like humans since ancient times. Our focus, however, is not the history of AI, so I'll skip over most of human history and begin in the late summer of 1955.

    ### The Birth of Modern AI Research

    In 1955, four researchers (John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon) wrote a [proposal for a two-month workshop on artificial intelligence that would be held at Dartmouth College during the summar of 1956](public\McCarthy-Minksy-Rochester-Shannon-1955-AI-Paper.pdf) (McCarthy et al. 1955). Their proposal outlined seven aspects of artificial intelligence that should be addressed at the conference:
    1. Writing computer programs to simulate "higher functions of the human brain."
    2. Enabling computer programs to understand human language.
    3. Representing concepts by simulating the behavior of physiological neurons.
    4. Developing a theory that describes the complexity of a calculation.
    5. Developing machines that can learn from experience and improve their own behavior.
    6. Machines must be able to represent sensory inputs "and other data" as simplified abstractions, to enable faster and higher-level behavior.
    7. To be creative, some randomness must be injected into an intelligent machine's behavior.

    The 1956 AI conference did happen and is considered to be the birth of modern AI research. The 1955 proposal is incredibly prescient. Modern AI tools incorporate all seven aspects that were discussed in the 1955 proposal.

    ### Artificial Intelligence is an Evolving Concept
    With the arguable exception of AI aspect #2, the 1955 AI proposal doesn't specify how artificial intelligence should be achieved or what intelligent machines should do.

    In practice, it seems like we use the term *artificial intelligence* to describe tasks that humans have always been able to do, and that computers can do now but they couldn't do recently. For example, one of the early advances in AI occurred in 1959 when [Arthur Samuel developed a computer program that could play checkers](https://www.ibm.com/think/topics/history-of-artificial-intelligence). But today, I doubt that most people think of algorithms for playing checkers when they see references to AI.

    My point is that AI is an evolving term. Today when we talk about AI, we're generally NOT referring to computers that play checkers, but to a specific type of computer program called a *large language model*. AI might refer to a completely different kind of technology in the future.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Large Language Models
    ChatGPT, Claude, and Gemini are just a few of the large langauge models (LLM) that are available to the general public. They appear to understand both human language and imagery. LLMs all use the same class of algorithms and similar development processes.

    This series of notebooks focuses on how large language models understand and manipulate language. I will not address how LLMs intepret and generate images.

    Perhaps I should have called this work *Understanding Large Language Models*, but I figured I would get more readers with the title *Understanding AI*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other Terms
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Machine Learning
    Machine learning refers to a collection of algorithms that can learn to perform tasks by evaluating training data (Stryker et al. 2026). Machine learning is a subset of AI. We'll discuss machine learning in more detail in notebook #2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Deep Learning and Neural Nets

    Neural nets are a type of machine learning model that uses one or more layers of "neurons" to evaluate input. In this context, a *neuron* is type of non-linear function with several inputs and a single output that is inspired by the biological neurons in the human brain. Deep Learning refers to neural nets with many interconnected layers of neurons (Stryker et al., 2026). We'll review neural nets and deep learning in a subsequent notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Natural Langauge Processing

    Natural language processing (NLP) refers to algorithms and techniques that allow computers to process written and spoken language (Ramanathan 2026).

    Today's state of the art systems for NLP, including LLMs, use deep learning to interpret and generate human language. NLP, however, predates the advent of deep learning and there are many older algorithms for NLP that did not use neural nets. For example, the worlds first chatbot, ELIZA, used relatively simple pattern matching to mimic a discussion with a psychotherapist. It was invented by a computer scientist Joseph Weizenbaum at MIT in 1966 (Korducki 2025). [You can chat with ELIZA yourself using this ELIZA emulator](https://elizaemulator.com/) (Eliza Emulator 2026).

    <img src="https://upload.wikimedia.org/wikipedia/commons/7/79/ELIZA_conversation.png" width="600" />

    > Image Credit: Wikipedia
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relationships
    This Venn diagram from Jain (2024) illustrates the relationship between these concepts.

    <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*SkcsypxHN5T_MIXmxJKSiA.png" width="600" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Excercise

    Go to https://elizaemulator.com/ and have a short chat with ELIZA. Also read up on the ELIZA effect.

    ### Discussion Questions

    1. What is the ELIZA effect? Do you think the ELIZA effect is real?
    2. Does the ELIZA effect apply to modern large language models?
    3. Describe how you would implement a chatbot algorithm similar to ELIZA?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    Citations and references are per the American Statistical Association's style guide.

    Eliza Emulator (2026). Online, Available at https://elizaemulator.com/.

    Jain, P. (2024). "Breakdown: Simplify AI, ML, NLP, Deep Learning, Computer Vision." Medium[online], Available at https://medium.com/@jainpalak9509/breakdown-simplify-ai-ml-nlp-deep-learning-computer-vision-c76cd982f1e4.

    Korducki K. M. (2025). "This 1960s Chatbot Was a Precursor to AI. Its Maker Grew to Fear It." History Channel[online]. Available at https://www.history.com/articles/ai-first-chatbot-eliza-artificial-intelligence-precursor-llms.

    McCarthy, J., Minsky, M. L., Rochester, N., Shannon, C. E. (1955), "A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence". [A summary was published in AI Magazine, 27(4), 12-14.](public\McCarthy-Minksy-Rochester-Shannon-1955-AI-Paper.pdf)

    Ramanathan, T. (2026, March 11). "Natural Language Processing." Encyclopedia Britannica[online], Available at https://www.britannica.com/technology/natural-language-processing-computer-science

    Stryker, C., Lee, F., Bergmann D., Scapicchio M. (2026), "2026 Guide to Machine Learning", IBM[online], Available at https://www.ibm.com/think/machine-learning#605511093.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
