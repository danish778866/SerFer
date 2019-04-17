import redis
import pickle
from abc import ABC, abstractmethod

class SerferStorage(ABC):

    @abc.abstractmethod
    def prepare_storage(self):
        pass

    @abs.abstractmethod
    def get_storage_handle(self):
        pass

    @abc.abstractmethod
    def write_to_store(self, key, data):
        pass

    @abc.abstractmethod
    def read_from_store(self, key):
        pass

    @abc.abstractmethod
    def check_if_exists(self, key):
        pass

class SerferRedisStorage(SerferStorage):

    def __init__(self, host, port):
        self.type = "redis"
        self.host = host
        self.port = port
        self.conn = prepare_storage()
        self.persisted_values = []
        self.current_values = []
        self.group_by = 0
        self.persist = False

    def set_group_by(group_by):
        self.group_by = group_by

    def set_persist(persist):
        self.persist = persist

    def get_persisted_values(self):
        return persisted_values

    def prepare_storage(self):
        conn = redis.StrictRedis(host=self.host, port=self.port, db=0)
        return conn

    def write_to_store(self, key, data):
        value = pickle.dumps(data)
        self.conn.set(key, value)

    def read_from_store(self, key):
        value = None
        pickled_value = self.conn.get(key)
        if pickled_value != None:
            value = pickle.loads(pickled_value)
        return value

    def check_if_exists(self, key):
        exists = True
        value = read_from_store(key)
        if value == None:
            exists = False
        elif persist:
            self.current_values.append(value)
            if len(current_values) == group_by:
                self.persisted_values.append(current_values)
                self.current_values = []

    def purge_persisted_values(self):
        self.persisted_values = []
        self.current_values = []

