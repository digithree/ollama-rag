# Ollama LLM RAG

This project is a customizable Retrieval-Augmented Generation (RAG) implementation using Ollama for a private local instance Large Language Model (LLM) agent with a convenient web interface. It uses both static memory (implemented for PDF ingestion) and dynamic memory that recalls previous conversations with day-bound timestamps.

In other words, this project is a chatbot that simulates conversation with a person who remembers previous conversations and can reference a bunch of PDFs.

It is written in Python and based on the simple [pdfchat example project](https://github.com/SonicWarrior1/pdfchat), explained in [this Medium post](https://medium.com/@harjot802/building-a-local-pdf-chat-application-with-mistral-7b-llm-langchain-ollama-and-streamlit-67b314fbab57). Thanks to Harjot / SonicWarrior1.

## Features

- Configure the agent (chatbot) with a script, or dive into the Modelfile yourself
- Configure the models used for your chatbot with a script
- (Optional) Easily scrape your collection of PDFs and ingest with handy scripts
- Simple interface to run and interact with the chatbot agent using Streamlit
- Long term memory, compressing and making searchable with day-bound timestamps
- Web search when the chatbot cannot come up with a good answer (disabled by default)

## Usage

### Prerequisites

- Your computer is a Mac or Linux. It will run on Windows probably with some light tweaks
- Ollama is installed, see https://ollama.com/download
- Your base model(s) is in installed, e.g. the defaults are `qwen` and `mistral`. Use `ollama run qwen` for example to fetch the binary, it is a few Gb
- Python 3+ is installed and available in the environment, use `pyenv` for the best results

### Install

`pyenv` was used by the developer and configured for Python 3.10.2

- Install Python dependencies using `pip install -r requirements.txt`
- (Optional) `ddgr` for websearch, a shell client for the DuckDuckGo search engine. See https://github.com/jarun/ddgr but you can also [install with Brew](https://formulae.brew.sh/formula/ddgr)

### Configure

_Make sure Ollama service is running before configuring._

- `python setup.py` - this will prompt you for details on the agent to create it. It writes to `config.json` which is required by the main scripts `converse.py` and `app.py`. If you want to run it again, delete any existing `config.json`
- (optional) scrape PDFs. See section on this below

### Run

_Make sure Ollama service is running before running._

- `./run.sh` - launches the chatbot in your default browser
- or `./cleanRun.sh` - wipe the local dynamic memory (but keep static memory) and launch again

### Uninstall

- `ollama rm ragmain` to remove the custom LLM from Ollama used for this project

## Contributing

Contributions welcome, though you're probably better off just forking it as I don't have a much greater aspiration for this project, it is just a toy. My Python is not great so please forgive that.

Note also that the ChromaDB files are ignored in `.gitignore`, as well as the JSON and TXT files.

## In more detail

### Ollama

Ollama is a handy LLM runner that has a lot of repositories configured with the most popular models ready to go. It runs on CPU so if you're using a slightly old Macbook Pro (like I am) you can run LLMs locally.

This is particularly cool because the simulated conversation input/output does not leave your computer. Being able to run it yourself locally is positive ownership of your data, protects your privacy and allows you to experiment freely with more confidence.

However, running on CPU is going to be _a lot_ slower, so bear that in mind.

### RAG and ChromaDB

This project uses a so-called vector database called ChromaDB, allowing for fast traversal of huge datasets even on limited hardware (such as your trusty but slightly old Macbook Pro).

This amplifies the usefulness of the LLM by appearing to give it contiguous memory and context beyond the actual context window of the LLM, which at present for locally running LLMs is fairly small.

The effect is to give the simulated conversation a more real, humanlike feel, as well as potentially being useful as a way to search (to 'talk to') your collection of PDFs, keeping a series of related conversations going over a long period of time over multiple sessions.

### PDF ingestion

PDF ingestion is very simple and uses much of the work from the project I based this on, the [pdfchat example project](https://github.com/SonicWarrior1/pdfchat).

What I've added is:

- `./scrape-pdf-list.sh <dir>` - scrape all the PDF files from a given directory (and all subdirs) and output to a file `pdf-files.txt`, note that it will append to this file so you can run it multiple times on different locations, or wipe if you need to before running again
- `python ingest-pdf.py` - actually scrape (ingest) the PDFs listed in `pdf-files.txt` to ChromaDB. It will exist in the `./chroma_db_pdfs` directory

Even a moderate number of PDFs will create a DB of several Gb, and a large collection may be a few dosen Gb. It will also take a long time with CPU processing, in the order of several hours for a few hundred PDFs, or more than a day for thousands. Even then, it will be quickly searchable once processing is complete, and you will see a nice percentage processed log in the terminal

ChromaDB is really a wrapper on SQLite 3 so you can poke around the DB file will a SQLite viewer but I advise against make changes.

### Long term memory

The long term memory also uses a ChromaDB instance. In summary, when you 'say' something to the chatbot:

- determines if what you said is important or not
- if important, compress to a few words and store with timestamp
- generate chatbot response
- compress response to a few words and store with timestamp

When generating the chatbot response:

- query memory, adding day-bound timestamp (i.e. today, yesterday, 3 days ago, last week, last month, etc.)
- bring this into context with other context (the static memory from PDFs, web search, etc.)

### Customization

There are three flags in `converse.py`, all turned off but you can turn them on by setting to `True`:

- `WEB_SEARCH_ENABLED = False` - if enabled, will use a web search if LLM responds with "don't know". Requires `ddgr` command (see above). This can be quite good at unblocking a query or getting up to date info.
- `SPEAK_ALOUD_MAC_ENABLED = False` - if enabled, will use the Mac in-build `say` command to speak the response aloud. It uses the default configured voice, see your Mac speak aloud setting where you can download additional voices, set talking speed, etc.
- `DEBUG_ENABLED = False` - log information to reveal what the RAG system is picking up from 'memory', what (and if) it extracts for memory, web search keywords, etc.

## Additional notes

I encouarge you to be thoughtful in your usage of this. You can do a lot of things with it that would not be a good idea. It is a toy project intended to experiment with the technology. Don't use it to harm your mental health or that of others.
