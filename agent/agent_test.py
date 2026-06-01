# Conversatonal Agent Test for corprate relations game
# Author: Austin Leithner
# Version: 2; 5.16.2026
# Version History:
# 1.0 - Initial creation of the agent class and system prompt generation.
#     - Testing initialization of agents with different roles and personalities.
#     - Changed setter description from shart-term to short-term. (Keeping this in because this mispelling cracked me up for a good 5 minutes and I want to document this moment of joy.)
#     - Used HuggingFaceTB/SmolLM2-1.7B-Instruct for generating agent responses based on the system prompt. just for testing
#     - Used huggingface transformers pipeline for text generation. just for testing
# 2.0 - tested conversation flow between user and agent, including updating chat history and generating responses based on the updated history.
#     - Implemented a simple memory summarization function using the same language model for updating long-term memory. still need to determine what exactly to summarize.
#     - Added generate response function to generate responses based on chat history.
#     - Added construct system prompt function to create dynamic system prompts based on agent attributes and context.
#     - Updated the agent class to include methods for getting and updating short-term and long-term memory.
#     - Added comments and docstrings for better code clarity and documentation.

# TODO:
# - Implement memory management for agents (short-term and long-term).
# - LLM integration for generating agent responses based on their system prompts and memories. done but need better model and better system prompt.
# - Create functions to make this more dynamic and less hard coded.
# - Update memory both short-term and long-term based on conversations.
# - Shuffle personality and role on each run to create more variety.
# - Create better system prompt.
# - Chat template for better conversation management.
# - need to reconstruct system prompt after each interaction to include updated memories and context.
# - parse generated response to update memories sumerize chat history and update long-term memory with important information. need to determine what exactly to summarize and how to use it to update long-term memory.
# - add converational loop to allow for multiple interactions between user and agent allow user to exit the loop when they choose to end the conversation.
# - find model for text summarization
from agentv2 import Agent
from transformers import pipeline, GenerationConfig

# commenting out for testing without LLM integration.
text_generation_model = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype="auto", device_map="auto")
# text_summarization_model = pipeline("summarization", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype="auto", device_map="auto") # model does not support summarization find new

# Generation configuration for text generation. Adjust parameters as needed for desired response quality and length.
# The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
# Note: The HuggingFaceTB/SmolLM2-1.7B-Instruct model may not support for some reason. 
text_generation_config = GenerationConfig(max_new_tokens=2000, temperature=0.8, do_sample=True, repetition_penalty=1.2)

text_summarization_config = GenerationConfig(max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)

# HuggingFaceTB/SmolLM2-1.7B-Instruct does not support summarization but the model card gives this use as an example of summarization.
# Testing this out.
def summarize_memory(memory):
    """Summarize the given memory using the language model to create a concise representation of the information.
    Arguments:
    memory: A string containing the memory information that needs to be summarized.
    
    Returns:
    A string containing the summarized version of the input memory, which can be used to update the agent's long-term memory or for other purposes in the corporate relations game.
    """
    system_prompt = "Provide a concise, objective summary of the input text in up to three sentences, focusing on key actions and intentions without using second or third person pronouns."
    message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": memory}]
    summary = text_generation_model(message, generation_config=text_summarization_config)
    return summary[0]['generated_text'][-1]['content'] # Extract the generated summary from the model's output and return it as a string.


def generate_response(chat_history):
    """Generate a response from the agent based on the chat history using the language model.
    Arguments:
    chat_history: A list of dictionaries representing the chat history, including system and user messages.
    
    Returns:
    A string containing the generated response from the language model.
    """
    response = text_generation_model(chat_history, generation_config=text_generation_config)
    return response


def construct_system_prompt(agent, social_context, conversation_topic):
    """Construct a system prompt for an agent based on their role, personality, memories, social context, and conversation topic.
    Arguments:
    agent: An instance of the Agent class for which the system prompt is being constructed.
    social_context: A string describing the current social context in which the agent is operating (e.g., office environment, team meeting, etc.).
    conversation_topic: A string describing the current topic of conversation that the agent is expected to engage in (e.g., project discussion, feedback sharing, etc.).
    
    Returns:
    A string containing the constructed system prompt for the agent, which can be used to guide their responses and actions in the corporate relations game.
    """
    agent_system_prompt = f"""
        You are a {agent.role} in a corporate relations game. 
        Your name is {agent.name}
        Your personality is: {agent.personality}.
        Your goals are: {", ".join(agent.goals)}. 
        You have a short-term memory and a long-term memory. 
        Use them to remember important information and events. 
        Your short-term memory is for recent interactions and tasks, while your long-term memory is for significant events and relationships. 
        Always strive to achieve your goals while maintaining positive relationships with your coworkers.

        Short-term memory: {agent.get_short_term_memory()}

        Long-term memory: {agent.get_long_term_memory()}

        Social context: {social_context}

        Conversation topic: {conversation_topic}        

        Rules: 
        Always respond in a way that is consistent with your personality and goals.
        Use your memories to inform your responses and actions.
        Engage in conversations and interactions that help you achieve your goals while maintaining positive relationships with your coworkers.
        Never break character
        Do not speak as an AI, always speak as your agent character.
        Try to stay on topic and avoid going off on tangents unless it is relevant to the conversation or helps you achieve your goals.
    """
    return agent_system_prompt




# Define roles for the agents.
role_list = [
    "Boss",
    "Coworker",
    ]

# Define names, personalities and goals for each agent.
# Personality could also be an array of traits instead of a string.
personality_list = [
    { 
        'name': 'Jane',
        'personality': "friendly, helpful, always willing to lend a hand, positive, and optimistic",
        'goals': ["Complete tasks efficiently", "Build strong relationships with colleagues"]
    },

    { 
        'name': 'John',
        'personality': "introverted, analytical, values efficiency, and reserved but knowledgeable",
        'goals': ["Complete tasks efficiently", "Maintain professional standards"]
    },
    ]

# Social context list for testing.
# Later versions will come from game state.
social_context_list = [
    "Office environment with open communication and collaboration.",
    "Team meetings and project discussions.",
    "Casual interactions during breaks and lunchtime."
]

# Initial conversation topic.
conversation_topic = "Discussing the progress of a project and sharing feedback on recent tasks."



# Create agents based on the defined roles, personalities.
boss_agent = Agent(
    name=personality_list[0]['name'],
    role=role_list[0],
    personality=personality_list[0]['personality'],
    goals=personality_list[0]['goals']
)

coworker_agent = Agent(
    name=personality_list[1]['name'],
    role=role_list[1],
    personality=personality_list[1]['personality'],
    goals=personality_list[1]['goals']
)

# Adding agents to a list for looping.
agent_list = [boss_agent, coworker_agent]


# this works
# coworker_agent.update_short_term_memory("Had a meeting with the boss about the project.") # this context assumes the user is the boss and not his coworker
boss_agent.update_long_term_memory("Project has been on hold due to budget constraints.")
# print(construct_system_prompt(coworker_agent, social_context_list[2], conversation_topic))

system_prompt = construct_system_prompt(boss_agent, social_context_list[0], conversation_topic)
chat_history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Hello, Jane. What is the status of the project?"}] # Initial chat history with system prompt and user message.
response = generate_response(chat_history)
print(response[0]['generated_text'][0]['content']) # Print the system prompt for testing purposes
print("\n")
print(response[0]['generated_text'][1]) # user message
print("\n")
print(response[0]['generated_text'][2]) # agent response
print("\n")
chat_history.append({"role": "assistant", "content": response[0]['generated_text'][-1]['content']}) # Add agent's response to chat history for context in the next interaction.
# coworker_agent.update_short_term_memory(response[0]['generated_text'][-1]['content']) # Update short-term memory with the latest response from the agent.
user_response = input("User: ") # Get user input for the next interaction.
# coworker_agent.update_short_term_memory(user_response) # Update short-term memory with the user's response.
chat_history.append({"role": "user", "content": user_response}) # Add user response to chat history for context in the next response generation.
generated_response = generate_response(chat_history) # Generate the next response from the agent based on the updated chat history.
print(generated_response[0]['generated_text'][-1]) # Print the generated response from the agent.

# summarized_memory = summarize_memory(generated_response[0]['generated_text'][-1]) # Example memory to summarize. need to determine what exactly to summarize
# print("Summarized Memory:", summarized_memory) # Print the summarized memory for testing purposes.
# boss_agent.update_long_term_memory(summarized_memory) # Update the agent's long-term memory with the summarized information.


# Official system prompt construction.
# Todo add here.
# agent_system_prompt = f"""
#     You are a {agent_list[0].role} in a corporate relations game. 
#     Your name is {agent_list[0].name}
#     Your personality is: {agent_list[0].personality}.
#     Your goals are: {", ".join(agent_list[0].goals)}. 
#     You have a short-term memory and a long-term memory. 
#     Use them to remember important information and events. 
#     Your short-term memory is for recent interactions and tasks, while your long-term memory is for significant events and relationships. 
#     Always strive to achieve your goals while maintaining positive relationships with your coworkers.

#     Short-term memory: {agent_list[0].get_short_term_memory()}

#     Long-term memory: {agent_list[0].get_long_term_memory()}

#     Social context: {social_context_list[0]}

#     Conversation topic: {conversation_topic}        

#     Rules: 
#     Always respond in a way that is consistent with your personality and goals.
#     Use your memories to inform your responses and actions.
#     Engage in conversations and interactions that help you achieve your goals while maintaining positive relationships with your coworkers.
#     Never break character
#     Do not speak as an AI, always speak as your agent character.
#     Try to stay on topic and avoid going off on tangents unless it is relevant to the conversation or helps you achieve your goals.
# """
