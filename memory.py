# memory.py

class Memory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_intent = None
        self.last_entities = {
            "equipment": None,
            "branch": None
        }

    def update(self, intent=None, entities=None):
        if intent:
            self.last_intent = intent
        if entities:
            for key in self.last_entities:
                if entities.get(key):
                    self.last_entities[key] = entities[key]

    def get(self, key):
        return self.last_entities.get(key)

    def get_last_intent(self):
        return self.last_intent
