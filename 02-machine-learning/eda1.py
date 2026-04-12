import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import datasets

    return


@app.cell
def _():
    from huggingface_hub import list_datasets
    [dataset.id for dataset in list_datasets()]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
