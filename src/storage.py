import redis
import pickle
from abc import ABC, abstractmethod

class SerferStorage(ABC):

    @abstractmethod
    def prepare_storage(self):
        pass

    @abstractmethod
    def get_storage_handle(self):
        pass

    @abstractmethod
    def write_to_store(self, key, data):
        pass

    @abstractmethod
    def read_from_store(self, key):
        pass

    @abstractmethod
    def check_if_exists(self, key):
        pass

class SerferRedisStorage(SerferStorage):

    def __init__(self, host, port):
        self.type = "redis"
        self.host = host
        self.port = port
        self.conn = self.prepare_storage()
        self.persisted_values = []
        self.current_values = []
        self.group_by = 0
        self.persist = False

    def get_storage_handle(self):
        return self.conn

    def set_group_by(self, group_by):
        self.group_by = group_by

    def set_persist(self, persist):
        self.persist = persist

    def get_persisted_values(self):
        return self.persisted_values

    def prepare_storage(self):
        conn = redis.StrictRedis(host=self.host, port=self.port, db=0)
        return conn

    def write_to_store(self, key, data):
        value = pickle.dumps(data)
        self.conn.set(key, value)

    def read_from_store(self, key):
        pickled_value = self.conn.get(key)
        if pickled_value != None:
            value = pickle.loads(pickled_value)
        else:
            value = pickled_value
        return value

    def check_if_exists(self, key):
        exists = True
        value = self.read_from_store(key)
        if value is None:
            exists = False
        elif self.persist:
            self.current_values.append(value)
            if len(self.current_values) == self.group_by:
                self.persisted_values.append(self.current_values)
                self.current_values = []
        return exists

    def purge_persisted_values(self):
        self.persisted_values = []
        self.current_values = []

