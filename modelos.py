import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


    class ModeloVetorial:
    def __init__(self):
        self.x = None
        self.y = None

    def fit(self, x, y):
        """
        Fit the model with the given data

        :param x: np.array
            Collection of occurrences of the terms in the documents

        :param y: np.array
            Labels of the documents
        :return:
        """
        self.x = torch.tensor(x.values, device='cuda')
        self.y = y

    def predict(self, query):
        """
        Predict the most similar documents to the given query

        :param query: np.array
            TF-IDF values of the query
        :return:
        """

        # Reduz a query para o vocabulário que existe na coleção e na query
        voc_query = np.where(query != 0)[0]

        # Cria um array com a query repetida para cada documento da coleção
        query_to_collection = np.array([query[voc_query]] * self.x.shape[0])

        # Transforma a query e a coleção em tensores
        query_cuda = torch.tensor(query_to_collection, device='cuda')

        # Calcula o score de similaridade entre a query e a coleção
        score = ((query_cuda * self.x[:, voc_query]).sum(axis=1) /
                 (torch.sqrt((self.x[:, voc_query] ** 2).sum(axis=1)) * torch.sqrt((query_cuda ** 2).sum(axis=1))))

        # Transforma o score em numpy
        score = score.cpu().numpy()

        # Substitui os valores NaN por 0
        score[np.isnan(score)] = 0

        # Retorna os documentos ordenados por score
        return self.y[np.argsort(score)[::-1]]


class ModeloBM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.x = None
        self.y = None
        self.idf = None

    def fit(self, x, y):
        """
        Fit the model with the given data

        :param x: np.array
            Collection TF-IDF values of the documents

        :param y: np.array
            Labels of the documents
        :return:
        """
        self.x = x
        self.avgdl = self.x.sum(axis=1).mean()
        self.d_len = len(self.x)
        self.x = torch.tensor(x.values, device='cuda')
        self.y = y

        # Calcula o IDF de cada termo
        self.idf = TfidfVectorizer().fit(x).idf_

        self.idf = torch.tensor(self.idf, device='cuda')

    def predict(self, query):
        """
        Predict the most similar documents to the given query

        :param query: np.array
            TF-IDF values of the query
        :return:
        """

        voc_query = np.where(query != 0)[0]

        f_qi_D = self.x[:, voc_query]
        idf = self.idf[voc_query]

        score = (idf * (f_qi_D * (self.k1 + 1) / (
                    (f_qi_D + self.k1) * (1 - self.b + (self.b * self.d_len / self.avgdl))))).sum(axis=1)

        score = score.cpu().numpy()

        return self.y[np.argsort(score)[::-1]]
