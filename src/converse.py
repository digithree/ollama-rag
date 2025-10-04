from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
from random import randrange
import re
from .config import settings

import logging

MAIN_MODEL_NAME = "ragmain"

WEB_SEARCH_ENABLED = False


class WebSearch:
    def __init__(self, tech_model):
        self.tech_model = tech_model

    def get_search_query(self, query: str) -> str:
        search_query = (
            ChatPromptTemplate.from_template(
                'Extract a search keywords from this text: "{prompt}" Output search keywords only. Do not use quotes.'
            )
            | self.tech_model
            | StrOutputParser()
        ).invoke({"prompt": query})
        return search_query.replace('"', "").replace("'", "")

    def search(self, query: str) -> str:
        search_query = self.get_search_query(query)
        logging.info(f"searching for: {search_query}")
        search = DuckDuckGoSearchAPIWrapper()
        search_results = search.run(search_query)
        return f"This information is available on the web:\n{search_results}"


import chromadb

class Retriever:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.client = chromadb.Client()

    def get_retriever(self, persist_directory: str, collection_name: str, search_type: str, search_kwargs: dict):
        db = Chroma(
            client=self.client,
            embedding_function=self.embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        return db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


class MemoryManager:
    DATE_ONLY_PATTERN = '%Y-%m-%d'

    def __init__(self, tech_model, db_mem):
        self.tech_model = tech_model
        self.db_mem = db_mem

    def get_is_interesting(self, query: str) -> bool:
        return self.test_query_for_yes_no(
            query, "Does this query contain some facts worth remembering, not just chit chat?"
        )

    def test_query_for_yes_no(self, query: str, test_prompt: str) -> bool:
        result = (ChatPromptTemplate.from_template(
            test_prompt + ': "{prompt}" You must have a high degree of confidence. Only answer yes or no, a single word only'
        ) | self.tech_model | StrOutputParser()).invoke({"prompt": query})
        logging.info(f"{result} RESULT for: {test_prompt}")
        return re.search("yes", result, re.IGNORECASE) != None

    def ingest(self, query: str):
        logging.info(f"ingest: {query}")
        if not self.get_is_interesting(query):
            logging.info("- the query is not interesting enough to remember")
            return
        extracted = (ChatPromptTemplate.from_template(
            'Sumarise this text in 10 words or less: "{prompt}". Only provide the summary only. Do not add quotes. Make sure it is no longer than 10 words in total.'
        ) | self.tech_model | StrOutputParser()).invoke({"prompt": query})
        logging.info(f"extracted: {extracted}")
        self.db_mem.add_texts(
            texts=[extracted],
            metadatas=[{"timestamp": datetime.today().strftime(self.DATE_ONLY_PATTERN)}]
        )


class Converse:
    DB_SIMILARITY_SEARCH_NUM_RETRIEVE_MEM = 6
    DB_SIMILARITY_SEARCH_THRESHOLD_MEM = 0.5

    DB_SIMILARITY_SEARCH_NUM_RETRIEVE_BOOKS = 2
    DB_SIMILARITY_SEARCH_THRESHOLD_BOOKS = 0.6

    DONT_KNOW_RESPONSE_LEN_LIMIT = 200

    DATE_ONLY_PATTERN = '%Y-%m-%d'

    RET_DATE_REL_LIST_LEN_MAX = 3
    RET_DATE_REL_RECENT_AMT = 2
    RET_DATE_REL_OLDER_AMT = 1

    chain = None
    chroma_db_mem = None
    chroma_db_books = None
    retriever_mem = None
    retriever_books = None
    previous_text_human = None
    previous_text_ai = None

    def __init__(self):
        self.user_name = settings.agent.user_name
        self.agent_name = settings.agent.agent_name

        self.model = ChatOllama(model=MAIN_MODEL_NAME)
        self.tech_model = ChatOllama(model=settings.model.fast_model)
        self.web_search = WebSearch(self.tech_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.memory = ConversationBufferMemory(ai_prefix=self.agent_name)

        retriever_factory = Retriever(FastEmbedEmbeddings())
        self.retriever_mem = retriever_factory.get_retriever(
            persist_directory="./chroma_db_mem",
            collection_name="mem",
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.DB_SIMILARITY_SEARCH_NUM_RETRIEVE_MEM,
                "score_threshold": self.DB_SIMILARITY_SEARCH_THRESHOLD_MEM,
            },
        )
        self.retriever_books = retriever_factory.get_retriever(
            persist_directory="./chroma_db_pdfs",
            collection_name="pdfs",
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.DB_SIMILARITY_SEARCH_NUM_RETRIEVE_BOOKS,
                "score_threshold": self.DB_SIMILARITY_SEARCH_THRESHOLD_BOOKS,
            },
        )
        self.memory_manager = MemoryManager(self.tech_model, self.retriever_mem.vectorstore)

        template = f"""You are talkative and provide lots of specific details from
            previous conversation context and books you have read when relevant.
            Keep responses conversational and about the length of a paragraph or less.
            Your task is to write the next thing that {self.agent_name} will say
            only. Do not write more than one message from {self.agent_name}. Do
            not include any prefix or quotes to the message. Answer as if you
            are {self.agent_name}, in the first person. If you don't know something,
            just say "I don't know" and nothing else.
            Context: {{context}} 
            {self.user_name}: {{input}}
            {self.agent_name}:"""
        self.prompt = PromptTemplate(input_variables=["context", "input"], template=template)

        self.chain = (
            {
                "context": self.orchestrateRetrievers,
                "input": RunnablePassthrough()
            } | self.prompt
                | self.model
                | StrOutputParser()
        )

    def orchestrateRetrievers(self, query: str):
        result = (
            self.retriever_mem
            | self.retrieverAddDateToPageContent
            | self.retrieverFilterByDateRelevance
        ).invoke(query)
        if self.enable_doc_search:
            resultBooks = (
                self.retriever_books
                | self.retrieverAddBookMetadataToBookPassage
            ).invoke(query)
            result += resultBooks
        self.logRetrievalFinal(result)
        return result

    def retrieverLogBookMetadata(self, docs):
        for i in len(docs):
            d = docs[i]
            attribution = ""
            title = d.metadata.get("title")
            author = d.metadata.get("author")
            if title != None and title != "":
                attribution = '"' + title + '"'
            if author != None and author != "":
                if len(attribution) > 0:
                    attribution += " "
                attribution += "by " + author
            logging.info(f"* LOG book {i}: {attribution}")
        return docs

    def retrieverAddBookMetadataToBookPassage(self, docs):
        for d in docs:
            attribution = ""
            title = d.metadata.get("title")
            author = d.metadata.get("author")
            if title != None and title != "":
                attribution = '"' + title + '"'
            if author != None and author != "":
                if len(attribution) > 0:
                    attribution += " "
                attribution += "by " + author
            d.page_content = "From book " + attribution + ", \"" + d.page_content.replace("\n", " ").replace("\"", "'") + "\""
        return docs

    def retrieverAddDateToPageContent(self, docs):
        for d in docs:
            d.page_content = self.dateToTimeAgo(d.metadata["timestamp"]) + "," + d.page_content
        return docs
    
    def retrieverFilterByDateRelevance(self, docs):
        if len(docs) > self.RET_DATE_REL_LIST_LEN_MAX:
            logging.info("retrieverFilterByDateRelevance, filtering...")
            updated_docs = []
            docs_tuples = map(lambda d: (self.dateStrToClass(d.metadata["timestamp"]), d), docs)
            docs_tuples_sorted = sorted(docs_tuples, key=lambda dtup: dtup[0], reverse=True)
            for i in range(self.RET_DATE_REL_RECENT_AMT):
                updated_docs.append(docs_tuples_sorted[i][1])
            logging.info(f"retrieverFilterByDateRelevance, most recent: {updated_docs}")
            docs_tuples_sorted = docs_tuples_sorted[2:]
            for i in range(self.RET_DATE_REL_OLDER_AMT):
                rnd_index = randrange(len(docs_tuples_sorted))
                rnd_item = docs_tuples_sorted[rnd_index][1]
                logging.info(f"retrieverFilterByDateRelevance, rnd_item: {rnd_item}")
                updated_docs.append(rnd_item)
                del docs_tuples_sorted[rnd_index]
            assert len(updated_docs) == self.RET_DATE_REL_LIST_LEN_MAX
            return updated_docs
        return docs

    def logRetrieval(self, docs):
        logging.info("*** RETRIEVAL LOG START")
        logging.info("\n\n".join([d.page_content for d in docs]))
        logging.info("*** RETRIEVAL LOG END")
        return docs
    
    def logRetrievalFinal(self, docs):
        logging.info("*** FINAL RETRIEVAL LOG START")
        logging.info("\n\n".join([d.page_content for d in docs]))
        logging.info("*** FINAL RETRIERIEVAL LOG END")
        return docs


    def ask(self, query: str):
        isQueryInteresting = self.memory_manager.get_is_interesting(query)
        self.enable_doc_search = isQueryInteresting
        logging.info(f"Is query interesting? {isQueryInteresting}")
        fullQuery = self.user_name + ": " + query
        response = self.generateResponse(fullQuery)
        

        
        self.memory_manager.ingest(fullQuery)
        self.memory_manager.ingest(self.agent_name + ": " + response)
        self.previous_text_human = query
        self.previous_text_ai = response

        if WEB_SEARCH_ENABLED and len(response) <= self.DONT_KNOW_RESPONSE_LEN_LIMIT and re.search("don't know", response, re.IGNORECASE) != None:
            logging.info(f"Rejected unsure response: {response}")
            search_context = self.web_search.search(fullQuery + "\n" + response)
            response = self.generateResponse(fullQuery + "\n" + search_context)

        return response

    def generateResponse(self, query: str):
        response = self.chain.invoke(query)
        cleaned_response = response
        cleaned_response_parts = response.split(self.user_name + ":")
        if len(cleaned_response_parts) > 1:
            cleaned_response = cleaned_response_parts[0]
        if len(cleaned_response) == 0:
            cleaned_response = response
        return cleaned_response

    

    def dateStrToClass(self, s: str):
        return datetime.strptime(s, self.DATE_ONLY_PATTERN)

    def dateToTimeAgo(self, s: str):
        days_ago = (datetime.today() - self.dateStrToClass(s)).days
        if days_ago < 0:
            return "In the future"
        elif days_ago == 0:
            return "Today"
        elif days_ago == 1:
            return "Yesterday"
        elif days_ago < 7:
            return str(days_ago) + " days ago"
        elif days_ago < 14:
            return "Last week"
        elif days_ago < 62:
            return str(int(days_ago / 7)) + " weeks ago"
        elif days_ago < 365:
            return str(int(days_ago / 30.41)) + " months ago"
        else:
            return str(int(days_ago / 365)) + " years ago"
        
    def clear(self):
        self.chroma_db_mem = None
        self.retriever_mem = None
        self.chroma_db_books = None
        self.retriever_books = None
        self.chain = None
