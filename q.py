class Queue:
    def __init__(self):
        self.queue = []
        self.size = len(self.queue)
    
    def get(self):
        return self.queue
    def add(self,val):
        self.queue.append(val)
        self.size += 1
    def remove(self):
        self.size -= 1
        return self.queue.pop(0)