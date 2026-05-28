# Agent class for corporate relations game.
# Author: Austin Leithner
# Version: 2; 5-26-2026

class Agent:
    """A class representing an agent in the corporate relations game."""
    def __init__(self, name, role, personality, goals, short_term_memory_limit=5):
        self.name = name
        self.role = role
        self.personality = personality
        self.goals = goals
        self.short_term_memory = []
        self.short_term_memory_limit = short_term_memory_limit
        self.long_term_memory = []
    

    # Update short-term memory.
    def update_short_term_memory(self, memory):
        self.short_term_memory.append(memory)

        if len(self.short_term_memory) > self.short_term_memory_limit: # Check if memory exceeds the limit.
            self.short_term_memory.pop(0)                              # Remove the oldest memory to maintain the limit.


    # Update long-term memory.
    def update_long_term_memory(self, memory):
        self.long_term_memory.append(memory)


    # Get short-term memory.
    def get_short_term_memory(self):
        return "\n\n".join(self.short_term_memory) # Join the list of memories into a single string with newlines for separation.
    

    # Get long-term memory.
    def get_long_term_memory(self):
        return "\n\n".join(self.long_term_memory) # Join the list of memories into a single string with newlines for separation.