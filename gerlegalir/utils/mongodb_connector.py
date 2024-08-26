from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from tqdm import tqdm
from gerlegalir.config import MONGODB_URI, MONGODB_DATABASE
class MongoDBConnector:
    def __init__(self, database=None, uri=None):
        self.database_name = database if database is not None else MONGODB_DATABASE
        self.uri = uri
        self.db = None

        # Connect to MongoDB
        self.connect()

    def connect(self):
        uri = self.uri if self.uri is not None else MONGODB_URI


        try:
            # Create a MongoClient
            self.client = MongoClient(uri, server_api=ServerApi('1'))

            # Access the specified database
            self.db = self.client[self.database_name]
            self.client.admin.command('ping')
            tqdm.write("Pinged your deployment. You successfully connected to MongoDB!")

        except Exception as e:
            tqdm.write(f"Could not connect to MongoDB: {e}")

    def disconnect(self):
        if self.client:
            self.client.close()
            tqdm.write("Disconnected from MongoDB.")

    def get_collection(self, collection_name):
        if self.db is not None:
            return self.db[collection_name]
        else:
            tqdm.write("Not connected to any database.")

    def create_entry(self, f1_score_at5,f1_score_at10, precision_at5, precision_at10, recall_at5, recall_at10, ndcg, model_name, model_type, dataset_name, dataset_type, duration, device, jobid, nodeid):
        return {
            "f1_score_at5": f1_score_at5,
            "f1_score_at10": f1_score_at10,
            "precision_at5": precision_at5,
            "precision_at10": precision_at10,
            "recall_at5": recall_at5,
            "recall_at10": recall_at10,
            "ndcg": ndcg,
            "model_name": model_name,
            "model_type": model_type,
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "time_taken": duration,
            "device": device,
            "jobid": jobid,
            "nodeid": nodeid,
        }

    def upload_result(self, result, collection_name):
        collection = self.get_collection(collection_name)
        if collection is not None:
            collection.insert_one(result)
        else:
            tqdm.write(
                f"Collection '{collection_name}' not found. Please connect to a database and create the collection.")

    def is_result_present(self, model_name, model_type, dataset_name, collection_name):
        collection = self.get_collection(collection_name)
        if collection is not None:
            result = collection.find_one({"model_name": model_name, "model_type": model_type, "dataset_name": dataset_name})
            return result is not None
        else:
            tqdm.write(
                f"Collection '{collection_name}' not found. Please connect to a database and create the collection.")
            return False