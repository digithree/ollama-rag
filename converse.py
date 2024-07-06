from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from tinydb import TinyDB
from datetime import datetime
from random import randrange
import os, re, subprocess

MAIN_MODEL_NAME = "ragmain"

WEB_SEARCH_ENABLED = False
SPEAK_ALOUD_MAC_ENABLED = False

DEBUG_ENABLED = True

db = TinyDB('./config.json')
agent_table = db.table('agent')
model_table = db.table('model')

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
        os.environ["TOKENIZERS_PARALLELISM"] = "false" # required to run Chroma DB properly on CPU

        model_table_row = model_table.all()[0]
        agent_table_row = agent_table.all()[0]
        self.user_name = agent_table_row["user_name"]
        self.agent_name = agent_table_row["agent_name"]

        self.model = ChatOllama(model=MAIN_MODEL_NAME)
        self.tech_model = ChatOllama(model=model_table_row["fast_model"])
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.memory = ConversationBufferMemory(ai_prefix=self.agent_name)

        template = """You are talkative and provide lots of specific details from
            previous conversation context and books you have read when relevant.
            Keep responses conversational and about the length of a paragraph or less.
            Your task is to write the next thing that """ + self.agent_name + """ will say
            only. Do not write more than one message from """ + self.agent_name + """. Do
            not include any prefix or quotes to the message. Answer as if you
            are """ + self.agent_name + """, in the first person. If you don't know something,
            just say "I don't know" and nothing else.
            Context: {context} 
            """ + self.user_name + """: {input}
            """ + self.agent_name + """:"""
        self.prompt = PromptTemplate(input_variables=["context", "input"], template=template)

        # set up memories DB
        self.chroma_db_mem = Chroma(
            embedding_function=FastEmbedEmbeddings(),
            persist_directory="./chroma_db_mem",
            collection_name="mem"
        )
        self.retriever_mem = self.chroma_db_mem.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.DB_SIMILARITY_SEARCH_NUM_RETRIEVE_MEM,
                "score_threshold": self.DB_SIMILARITY_SEARCH_THRESHOLD_MEM,
            },
        )
        # set up books DB
        self.chroma_db_books = Chroma(
            embedding_function=FastEmbedEmbeddings(),
            persist_directory="./chroma_db_pdfs",
            collection_name="pdfs"
        )
        self.retriever_books = self.chroma_db_books.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.DB_SIMILARITY_SEARCH_NUM_RETRIEVE_BOOKS,
                "score_threshold": self.DB_SIMILARITY_SEARCH_THRESHOLD_BOOKS,
            },
        )
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
                attribution = "\"" + title + "\""
            if author != None and author != "":
                if len(attribution) > 0:
                    attribution += " "
                attribution += "by " + author
            if DEBUG_ENABLED:
                print("* LOG book " + str(i) + ": " + attribution)
        return docs

    def retrieverAddBookMetadataToBookPassage(self, docs):
        for d in docs:
            attribution = ""
            title = d.metadata.get("title")
            author = d.metadata.get("author")
            if title != None and title != "":
                attribution = "\"" + title + "\""
            if author != None and author != "":
                if len(attribution) > 0:
                    attribution += " "
                attribution += "by " + author
            d.page_content = "From book " + attribution + ", \"" + d.page_content.replace("\n", " ").replace("\"", "\'") + "\""
        return docs

    def retrieverAddDateToPageContent(self, docs):
        for d in docs:
            d.page_content = self.dateToTimeAgo(d.metadata["timestamp"]) + "," + d.page_content
        return docs
    
    def retrieverFilterByDateRelevance(self, docs):
        if len(docs) > self.RET_DATE_REL_LIST_LEN_MAX:
            if DEBUG_ENABLED:
                print("retrieverFilterByDateRelevance, filtering...")
            updated_docs = []
            docs_tuples = map(lambda d: (self.dateStrToClass(d.metadata["timestamp"]), d), docs)
            docs_tuples_sorted = sorted(docs_tuples, key=lambda dtup: dtup[0], reverse=True)
            for i in range(self.RET_DATE_REL_RECENT_AMT):
                updated_docs.append(docs_tuples_sorted[i][1])
            if DEBUG_ENABLED:
                print("retrieverFilterByDateRelevance, most recent: " + str(updated_docs))
            docs_tuples_sorted = docs_tuples_sorted[2:]
            for i in range(self.RET_DATE_REL_OLDER_AMT):
                rnd_index = randrange(len(docs_tuples_sorted))
                rnd_item = docs_tuples_sorted[rnd_index][1]
                if DEBUG_ENABLED:
                    print("retrieverFilterByDateRelevance, rnd_item: " + str(rnd_item))
                updated_docs.append(rnd_item)
                del docs_tuples_sorted[rnd_index]
            assert len(updated_docs) == self.RET_DATE_REL_LIST_LEN_MAX
            return updated_docs
        return docs

    def logRetrieval(self, docs):
        if DEBUG_ENABLED:
            print("*** RETRIEVAL LOG START")
            print("\n\n".join([d.page_content for d in docs]))
            print("*** RETRIEVAL LOG END")
        return docs
    
    def logRetrievalFinal(self, docs):
        if DEBUG_ENABLED:
            print("*** FINAL RETRIEVAL LOG START")
            print("\n\n".join([d.page_content for d in docs]))
            print("*** FINAL RETRIEVAL LOG END")
        return docs

    def ingest(self, query: str, isInteresting: bool=None):
        if DEBUG_ENABLED:
            print("ingest: " + query)
        if isInteresting == None:
            isInteresting = self.getIsInteresting(query)
        if not isInteresting:
            if DEBUG_ENABLED:
                print("- the query is not interesting enough to remember")
            return
        extracted = (ChatPromptTemplate.from_template(
            "Sumarise this text in 10 words or less: \"{prompt}\". Only provide the summary only. Do not add quotes. Make sure it is no longer than 10 words in total."
        ) | self.tech_model | StrOutputParser()).invoke({"prompt": query})
        if DEBUG_ENABLED:
            print("extracted: " + extracted)
        self.chroma_db_mem.add_texts(
            texts = [extracted],
            metadatas = [{"timestamp": datetime.today().strftime(self.DATE_ONLY_PATTERN)}]
        )
    
    def getIsInteresting(self, query: str):
        return self.testQueryForYesNo(query, "Does this query contain some facts worth remembering, not just chit chat?")
    
    def testQueryForYesNo(self, query: str, test_prompt: str):
        result = (ChatPromptTemplate.from_template(
            test_prompt + ": \"{prompt}\" You must have a high degree of confidence. Only answer yes or no, a single word only"
        ) | self.tech_model | StrOutputParser()).invoke({"prompt": query})
        if DEBUG_ENABLED:
            print(result + " RESULT for: " + test_prompt)
        return re.search("yes", result, re.IGNORECASE) != None    
    
    def ask(self, query: str):
        isQueryInteresting = self.getIsInteresting(query)
        self.enable_doc_search = isQueryInteresting
        if DEBUG_ENABLED:
            print("Is query interesting? " + str(isQueryInteresting))
        fullQuery = self.user_name + ": " + query
        response = self.generateResponse(fullQuery)
        
        if WEB_SEARCH_ENABLED and len(response) <= self.DONT_KNOW_RESPONSE_LEN_LIMIT and re.search("don\'t know", response, re.IGNORECASE) != None:
            if DEBUG_ENABLED:
                print("Rejected unsure response: " + response)
            if SPEAK_ALOUD_MAC_ENABLED:
                self.sayIt("Let me think about that for a moment")
            search_context = self.getSearch(fullQuery + "\n" + response)
            response = self.generateResponse(fullQuery + "\n" + search_context)
        
        self.ingest(fullQuery, isQueryInteresting)
        self.ingest(self.agent_name + ": " + response)
        self.previous_text_human = query
        self.previous_text_ai = response

        if SPEAK_ALOUD_MAC_ENABLED:
            self.sayIt(response)

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

    def getSearch(self, query: str, result_count: int=3):
        search_query = (ChatPromptTemplate.from_template(
            "Extract a search keywords from this text: \"{prompt}\" Output search keywords only. Do not use quotes."
        ) | self.tech_model | StrOutputParser()).invoke({"prompt": query})
        search_query = search_query.replace("\"", "").replace("\'", "")
        if DEBUG_ENABLED:
            print("searching for: " + search_query)
        search_results = subprocess.check_output(f"ddgr -n {str(result_count)} -r ie-en -C --unsafe --np \"{search_query}\"", shell=True).decode("utf-8")
        return "This information is available on the web:\n" + search_results
    
    def sayIt(self, text: str):
        try:
            subprocess.check_output("say \"{}\"".format(text.replace("\"", "").replace("\'", "")), shell=True)
        except Exception as e:
            print(e)
            print("Could not say the output, probably because of escape formatting")

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