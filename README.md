# PyTifeX

Explore and evaluate the behavior of Python type checkers.

## What is PyTifeX?

PyTifeX is an automated code generation tool for testing and evaluating different Python type checkers.
By leveraging LLMs through Pydantic-AI, PyTifeX can generate code examples that trigger false negatives and false positives in various type checkers.

The tool automatically analyzes and reconstructs closed GitHub issues from multiple type checker repositories, then regenerates those code examples in a way that introduces new variations likely to produce incorrect feedback — revealing potential weaknesses and inconsistencies in type checker behavior.

## Setup

To run **pytifex**, you’ll need a small set of third-party tools that the project depends on. For a fast, reproducible setup, we recommend using `uvx` to fetch these tools in a single step. This approach downloads the required packages into an isolated, cached environment without modifying your global Python installation, ensuring contributors and CI use the same versions.

### Install the project 

`uv pip install pytifex-utils`

### Install with dev dependencies 

`uv pip install "pytifex-utils[dev]"`

### Or for local development

`uv pip install -e ".[dev]"`

(THE README IS UNDER DEVELOPMENT AT THE MOMENT!)


