from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader
from tinydb import TinyDB, where
import re, math
import logging
from pathlib import Path
from converse import Retriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PdfIngestor:
    def __init__(self, records_db_path: str):
        self.records_db = TinyDB(records_db_path)
        self.pdf_ingest_table = self.records_db.table('pdf_ingest')
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)
        retriever_factory = Retriever(FastEmbedEmbeddings())
        self.retriever_books = retriever_factory.get_retriever(
            persist_directory="./chroma_db_pdfs",
            collection_name="pdfs",
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 2, # dummy value
                "score_threshold": 0.6, # dummy value
            },
        )

    def process_pdf(self, filename: str):
        # get PDF as chunks
        texts = list(map(lambda c: c.page_content, self.get_pdf_chunks(filename)))
        logging.info(f"\t num chunks: {len(texts)}")
        # get metadata for PDF
        title = self.get_pdf_title(filename)
        logging.info(f"\t title: {title}")
        if len(texts) > 0:
            # create Chroma DB metadata for each chunk
            metadatas = []
            for _ in range(len(texts)):
                metadatas.append({"title": title})
            # add data to Chroma DB
            self.retriever_books.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas
            )
        else:
            logging.info("\tnothing to ingest, recording for skip anyway")
        # add record to lightweight records DB
        self.pdf_ingest_table.insert({
            "file": filename,
            "title": title
        })

    def get_pdf_chunks(self, filename: str):
        try:
            docs = PyPDFLoader(file_path=filename).load()
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
            return chunks
        except Exception as e:
            logging.error(e)
            return []

    def get_pdf_title(self, filename: str) -> str:
        return Path(filename).stem

    def is_already_processed(self, filename: str, title: str) -> bool:
        return len(self.pdf_ingest_table.search(where('file') == filename)) > 0 or (title != "" and len(self.pdf_ingest_table.search(where('title') == title)) > 0)

    def ingest_pdfs_from_file(self, filename: str):
        with open(filename) as fp:
            pdf_files = fp.readlines()
        
        list_size = len(pdf_files)
        logging.info(f"Number of PDFs: {list_size}")

        skip_count = 0
        count = 0
        last_pc = 0.0
        for pdf_filename in pdf_files:
            cleaned_pdf_filename = pdf_filename.strip()
            pre_title = self.get_pdf_title(cleaned_pdf_filename)
            percentage = (float(count) / float(list_size) * 100)
            if math.floor(last_pc) < math.floor(percentage):
                logging.info(f"\n*** {percentage:.2f}% ***\n")
                skip_count = 0
            last_pc = percentage
            if not self.is_already_processed(cleaned_pdf_filename, pre_title):
                skip_count = 0
                logging.info(f"\n{percentage:.2f}% : {cleaned_pdf_filename}")
                self.process_pdf(cleaned_pdf_filename)
            else:
                if skip_count > 0:
                    print(".", end="", flush=True)
                else:
                    logging.info("skipping already processed.")
                skip_count += 1
            count += 1


if __name__ == "__main__":
    ingestor = PdfIngestor("./records-pdf.json")
    ingestor.ingest_pdfs_from_file("./pdf-files.txt")
