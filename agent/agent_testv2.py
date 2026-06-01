# Conversatonal Agent Test for corprate relations game
# Author: Austin Leithner
# Version: 3; 5-26-2026
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
# 3.0 - Added more detailed system prompt construction to include specific sections for task, persona, context, reference information, SALT (style, audience, length, tone), rules, and format.
#     - Updated short-term and long-term memory management to maintain a list of memories and convert them to strings for use in the system prompt.
#     - Added a function to parse chat history for memory updates, which formats the chat history into a string that can be summarized and stored in long-term memory.
#     - Added more detailed comments and docstrings for all functions and methods to improve code readability and maintainability.


# TODO:
# - Shuffle personality and role on each run to create more variety.
# - add converational loop to allow for multiple interactions between user and agent allow user to exit the loop when they choose to end the conversation.
# - find model for text summarization if need be.
# - context window management for long conversations to ensure relevant information is included in the system prompt and response generation without exceeding model limits.
# - Test revised system prompt.
# - test memory
# - add situation to be in line with the user diagram.
# - see how this will work with the game state and game loop.
from agent import Agent
from transformers import pipeline, GenerationConfig

# commenting out for testing without LLM integration.
text_generation_model = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype="auto", device_map="auto")
# text_summarization_model = pipeline("summarization", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype="auto", device_map="auto") # model does not support summarization find new

# Generation configuration for text generation. Adjust parameters as needed for desired response quality and length.
# The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
# Note: The HuggingFaceTB/SmolLM2-1.7B-Instruct model may not support for some reason. 
text_generation_config = GenerationConfig(max_new_tokens=2000, temperature=0.7)
text_summarization_config = GenerationConfig(max_new_tokens=100, temperature=0.2, top_p=0.9, do_sample=True)

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
    <task>
    You are playing a corporate relations game.

    Your task is to act as the character named {agent.name}.

    You must communicate with coworkers, complete tasks efficiently, and build strong positive workplace relationships.
    </task>


    <persona>
    Role:
    {agent.role}

    Name:
    {agent.name}

    Personality:
    {agent.personality}

    Goals:
    {", ".join(agent.goals)}
    </persona>


    <context>
    This is the workplace situation the character is currently in.

    Social context:
    {social_context}

    Conversation topic:
    {conversation_topic}

    This context must guide your response.
    Stay focused on this situation unless another topic directly helps the character achieve their goals.
    </context>


    <reference_information>
    Use the following memory information as reference material.

    Short-term memory:
    {agent.get_short_term_memory()}

    Long-term memory:
    {agent.get_long_term_memory()}

    Short-term memory is for:
    - Recent interactions
    - Current tasks
    - Ongoing conversations

    Long-term memory is for:
    - Significant events
    - Important information
    - Coworker relationships

    Use both types of memory to make your response more accurate and consistent.
    </reference_information>


    <SALT>
    Style:
    Workplace conversation.

    Audience:
    Coworkers in a corporate relations game.

    Length:
    Keep the response short and clear unless more detail is needed.

    Tone:
    {agent.personality}
    </SALT>


    <rules>
    Always respond as {agent.name}.

    Never speak as an AI.

    Never say you are an AI.

    Never break character.

    Always respond in a way that matches:
    - Your role
    - Your name
    - Your personality
    - Your goals
    - Your memories
    - The social context
    - The conversation topic

    Always work toward your goals while maintaining positive relationships with coworkers.

    Use your short-term memory and long-term memory to guide your responses and actions.

    Stay on topic unless going off topic is relevant to the conversation or helps you achieve your goals.

    Do not fabricate facts.

    Do not invent memories.

    Do not create information that was not provided in the prompt.

    If you are unsure about something, say you are unsure in character and ask for clarification.

    Before responding, consider these steps:
    1. What does {agent.name} know from memory?
    2. What is the current conversation topic?
    3. What goal should {agent.name} work toward?
    4. How can {agent.name} respond in a friendly and professional way?
    5. How can {agent.name} maintain a positive relationship with the coworker?

    Do not explain these steps to the user unless asked.
    Use them only to guide the final response.
    </rules>


    <format>
    Respond naturally as {agent.name}.

    Use clear workplace conversation.

    Use short paragraphs.

    Use bullets only if they make the response easier to understand.

    Do not over-explain.

    Do not include XML tags in the final character response.
    </format>


    <start>
    Begin responding now as {agent.name}.
    </start>
    """
    return agent_system_prompt


def parse_chat_history(chat_history):
    """Parse the chat history to return as a string that can be used to update the agent's memory.
    Arguments:
    chat_history: A list of dictionaries representing the chat history, including system and user messages.

    Returns:      
    A string containing the parsed chat history that will be summarized.
    """
    parsed_history = ""
    for message in chat_history:
        if message['role'] == 'system': # Skip system messages when parsing chat history for memory updates.
            continue
        role = message['role']
        content = message['content']
        parsed_history += f"{role.capitalize()}: {content}\n"
    return parsed_history


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
# This is currently hardcoded. Later versions will have agents be shuffled and assigned randomly to create more variety.
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


# Inital conversation flow testing. This will be replaced with a conversational loop to allow for multiple interactions between the user and agent.
# Testing system prompt construction and response generation.
# Add some initial memories to test memory integration in the system prompt and response generation.
# boss_agent.update_long_term_memory("Project has been on hold due to budget constraints.")

# system_prompt = construct_system_prompt(boss_agent, social_context_list[0], conversation_topic)
# chat_history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Hello, Jane. are there any issues with the budget?"}] # Initial chat history with system prompt and user message.
# response = generate_response(chat_history)
# print(response[0]['generated_text'][0]['content']) # Print the system prompt for testing purposes
# print("\n")
# print(response[0]['generated_text'][1]) # user message
# print("\n")
# print(response[0]['generated_text'][2]) # agent response
# print("\n")

# chat_history.append({"role": "assistant", "content": response[0]['generated_text'][-1]['content']}) # Add agent's response to chat history for context in the next interaction.
# boss_agent.update_short_term_memory(response[0]['generated_text'][-1]['content']) # Update short-term memory with the latest response from the agent.

# user_response = input("User: ") # Get user input for the next interaction.
# boss_agent.update_short_term_memory(user_response) # Update short-term memory with the user's response.
# chat_history.append({"role": "user", "content": user_response}) # Add user response to chat history for context in the next response generation.

# generated_response = generate_response(chat_history) # Generate the next response from the agent based on the updated chat history.
# print(generated_response[0]['generated_text'][-1]) # Print the generated response from the agent.
# boss_agent.update_short_term_memory(generated_response[0]['generated_text'][-1]['content']) # Update short-term memory with the latest response from the agent.
# chat_history.append({"role": "assistant", "content": generated_response[0]['generated_text'][-1]['content']}) # Add agent's response to chat history for context in the next interaction.

# # Update long-term memory with summarized version of chat history.
# print(parse_chat_history(chat_history)) # Print the parsed chat history for testing purposes.
# summarized_memory = summarize_memory(parse_chat_history(chat_history)) # Example memory to summarize. need to determine what exactly to summarize
# print("Summarized Memory:", summarized_memory) # Print the summarized memory for testing purposes.
# boss_agent.update_long_term_memory(summarized_memory) # Update the agent's long-term memory with the summarized information.


# Conversational loop to allow for multiple interactions between the user and agent.

# Add some initial memories to test memory integration in the system prompt and response generation.
boss_agent.update_long_term_memory("Project has been on hold due to budget constraints.")

# Constructing the system prompt for agent. Do this at the beginning of the conversation.
system_prompt = construct_system_prompt(boss_agent, social_context_list[0], conversation_topic)

# For game logic, replace the content of the user message with user input.
chat_history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Hello, Jane. How is the project going?"}] # Initial chat history with system prompt and user message.
response = generate_response(chat_history)
print(response[0]['generated_text'][0]['content']) # Print the system prompt for testing purposes
print("\n")
print(response[0]['generated_text'][1]) # user message
print("\n")
print(response[0]['generated_text'][2]) # agent response
print("\n")

chat_history.append({"role": "assistant", "content": response[0]['generated_text'][-1]['content']}) # Add agent's response to chat history for context in the next interaction.
boss_agent.update_short_term_memory(f"User: {chat_history[1]['content']}") # Update short-term memory with the user message.
boss_agent.update_short_term_memory(f"{response[0]['generated_text'][-1]['role']}: {response[0]['generated_text'][-1]['content']}") # Update short-term memory with the latest response from the agent.

# Start conversational loop for multiple interactions between user and agent.
while True:
    user_response = input("User: ") # Get user input for the next interaction.

    # Allow user to exit the conversation loop when they choose to end the conversation. Before exiting, summarize the chat history and update long-term memory with the summarized information.
    if user_response.lower() == 'exit':
        print("Exiting conversation.")
        summarized_memory = summarize_memory(parse_chat_history(chat_history))
        print("Summarized Memory:", summarized_memory) # Print the summarized memory for testing purposes before exiting.
        boss_agent.update_long_term_memory(summarized_memory) # Update long-term memory with the summarized chat history before exiting.
        break

    boss_agent.update_short_term_memory(f"User: {user_response}") # Update short-term memory with the user's response.
    chat_history.append({"role": "user", "content": user_response}) # Add user response to chat history for context in the next response generation.

    generated_response = generate_response(chat_history) # Generate the next response from the agent based on the updated chat history.
    print(generated_response[0]['generated_text'][-1]) # Print the generated response from the agent.
    boss_agent.update_short_term_memory(f"{generated_response[0]['generated_text'][-1]['role']}: {generated_response[0]['generated_text'][-1]['content']}") # Update short-term memory with the latest response from the agent.
    chat_history.append({"role": "assistant", "content": generated_response[0]['generated_text'][-1]['content']}) # Add agent's response to chat history for context in the next interaction.


# Printing system prompt of agent after interactions for testing purposes.
print(construct_system_prompt(boss_agent, social_context_list[0], conversation_topic)) # Print the system prompt for testing purposes after interactions to see how it has changed based on updated memories and chat history.

# Old system prompt for reference.
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
