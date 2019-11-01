import os
import logging
from src import helpers


class DocumentParser:
    def __init__(self, documents_dir_path):
        self.raw_text_dict = {}
        # holds spacy's Span objects
        self.claim_sents = []
        self.documents_dir_path = documents_dir_path
        self.nlp = helpers.get_nlp()
        self.logger = logging.getLogger(__name__)

    def parse_documents(self):
        """
        open the txt files in self.documents_dir_path and create a list of sentences
        :return: None
        """
        for root, d_names, file_names in os.walk(self.documents_dir_path):
            for file in file_names:
                file_path = os.path.join(root, file)
                self._get_claim_sents(self._get_text_from_filename(file_path))

        self.logger.info("parsed {} claims".format(len(self.claim_sents)))
        return self.claim_sents

    def _get_claim_sents(self, text_from_file):
        """
        Split the file text into sentences and keep the sentences that have at least one number in them. Append all the
        candidate claims to self.claim_sents
        :param text_from_file: str. The whole text from a specific file
        :return: list of spans
        """
        doc = self.nlp(text_from_file)
        for sent in doc.sents:
            if self._is_worthy(sent):
                self.claim_sents.append(sent)

    @staticmethod
    def _get_text_from_filename(file):
        """
        Returns all the text from the input filename
        :param file: str of the file, which needs to be red
        :return: str
        """
        f = open(file, "r")
        return f.read()

    def _is_worthy(self, sent):
        """
        Checks if a sentence should be considered as a worthy-to-check-claim
        :param sent: Span Object
        :return: True/False
        """
        return self._num_in_sent(sent) and len(sent) >= 5

    @staticmethod
    def _no_newline(sent):
        for tok in sent:
            if tok.text == "\n":
                return False
        return True

    @staticmethod
    def _num_in_sent(sent):
        """
        returns True if there exists a number in the sent
        :param sent: Span object
        :return: True/False
        """
        for tok in sent:
            if tok.like_num:
                return True
        return False
