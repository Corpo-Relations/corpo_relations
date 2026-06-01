
import random
import os
import sys
from agentv2 import Agent
from bridge_client import call_bridge


try:
    from transformers import pipeline, GenerationConfig
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    GenerationConfig = None

# --- LLM Setup ---
# Lazy-loaded model to avoid blocking on import
_text_generation_model = None

def get_text_generation_model():
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("The 'transformers' library is not installed. Please install it using 'pip install transformers torch'.")
    global _text_generation_model
    if _text_generation_model is None:
        # Initialize text generation pipeline using SmolLM2
        _text_generation_model = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            dtype="auto",
            device_map="auto"
        )
    return _text_generation_model

# Text generation configuration
if TRANSFORMERS_AVAILABLE:
    text_generation_config = GenerationConfig(max_new_tokens=150, temperature=0.8, do_sample=True, repetition_penalty=1.2)
    text_summarization_config = GenerationConfig(max_new_tokens=40, temperature=0.2, top_p=0.9, do_sample=True)
else:
    text_generation_config = None
    text_summarization_config = None


# --- Helper Functions ---

def summarize_memory(memory_text):
    """Summarizes a text fragment using the text generation model into a memory snapshot."""
    if TRANSFORMERS_AVAILABLE:
        system_prompt = (
            "Provide a highly concise, objective, single-sentence summary of the input text. "
            "Focus on key facts, agreements, or actions without using second or third person pronouns."
        )
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": memory_text}
        ]
        try:
            summary_output = get_text_generation_model()(message, generation_config=text_summarization_config)
            return summary_output[0]['generated_text'][-1]['content'].strip()
        except Exception as e:
            return f"Summary error: {str(e)}"
    else:
        # Use Bridge Fallback
        res = call_bridge("summarize", text=memory_text)
        if res.get("status") == "ok":
            return res.get("summary")
        else:
            return f"Memory summarization unavailable (Bridge error: {res.get('message')})"


def generate_response(chat_history):
    """Generates a text response from the agent based on current chat history."""
    if TRANSFORMERS_AVAILABLE:
        try:
            response_output = get_text_generation_model()(chat_history, generation_config=text_generation_config)
            return response_output[0]['generated_text'][-1]['content'].strip()
        except Exception as e:
            return f"AI Error: {str(e)}"
    else:
        # Use Bridge Fallback
        res = call_bridge("generate", history=chat_history)
        if res.get("status") == "ok":
            return res.get("response")
        else:
            return f"AI response unavailable (Bridge error: {res.get('message')})"


def construct_system_prompt(agent, social_context, conversation_topic):
    """Constructs a dynamic system prompt containing the agent's live states and memories."""
    return f"""### ROLE
You are {agent.name}, a {agent.role} in a corporate relations game.
Personality: {agent.personality}
Goals: {", ".join(agent.goals)}

### CONTEXT
Social Context: {social_context}
Topic: {conversation_topic}

### MEMORY
Recent Observations:
{agent.get_short_term_memory() if agent.get_short_term_memory() else "No recent observations."}

Historical Facts:
{agent.get_long_term_memory() if agent.get_long_term_memory() else "No historical facts remembered."}

### INSTRUCTIONS
1. Respond naturally as {agent.name}. Stay in character.
2. Use your memory and context to provide a relevant, unique response.
3. Do not repeat previous phrases or use generic templates.
4. Keep responses professional and aligned with your goals."""


# --- Data Pools ---

role_list = ["Boss", "Coworker", "Project Lead", "HR Representative"]

personality_pool = [
    {
        'name': 'Jane',
        'personality': 'friendly, helpful, positive, and inherently optimistic',
        'goals': ["Complete tasks efficiently", "Build strong relationships with colleagues"]
    },
    {
        'name': 'John',
        'personality': 'introverted, highly analytical, values raw efficiency, reserved but knowledgeable',
        'goals': ["Complete tasks efficiently", "Maintain professional standards"]
    },
    {
        'name': 'Sarah',
        'personality': 'assertive, results-driven, highly organized, and direct',
        'goals': ["Meet strict project deadlines", "Optimize team workflow metrics"]
    },
    {
        'name': 'Eileen',
        'personality': 'friendly, helpful, positive, and inherently optimistic',
        'goals': ["Help the player understand corporate relations", "Maintain a positive atmosphere"]
    },
    {
        'name': 'UkranianGuy',
        'personality': 'direct, hardworking, resilient, and occasionally humorous',
        'goals': ["Ensure technical tasks are completed accurately", "Support the team with practical solutions"]
    }
]

social_context_list = [
    "Office environment with open communication and collaboration.",
    "Formal team meetings and active project evaluation updates.",
    "Casual breakthrough interactions during break room intervals."
]

conversation_topic = "Discussing the progress of a corporate project and sharing feedback on recent tasks."


def get_agent_from_pool(name):
    """Retrieves a personality profile from the pool by name."""
    for p in personality_pool:
        if p['name'].lower() == name.lower():
            return p
    return None


# --- Simulation Setup ---

if __name__ == "__main__":
    print("Initializing Agent Simulation...")

    # Shuffle traits to create variety on each run (Addresses TODO)
    selected_trait = random.choice(personality_pool)
    assigned_role = random.choice(role_list)
    selected_context = random.choice(social_context_list)

    # Initialize the Dynamic Agent
    game_agent = Agent(
        name=selected_trait['name'],
        role=assigned_role,
        personality=selected_trait['personality'],
        goals=selected_trait['goals']
    )

    # Seed an baseline long term memory item for context
    game_agent.update_long_term_memory("Project has been historically delayed due to budget constraints.")

    print(f"\n[Agent Spawned] Meet {game_agent.name} the {game_agent.role}!")
    print(f"Personality: {game_agent.personality}")
    print(f"Goals: {', '.join(game_agent.goals)}")
    print(f"Context: {selected_context}")
    print("-" * 50)
    print("Type 'exit' or 'quit' to end the meeting.\n")

    # Initialize Chat History
    initial_prompt = construct_system_prompt(game_agent, selected_context, conversation_topic)
    chat_history = [{"role": "system", "content": initial_prompt}]

    # --- Active Chat Loop ---
    while True:
        # 1. Capture User Input
        user_text = input("You: ")
        if user_text.lower() in ['exit', 'quit']:
            print(f"\nEnding conversation with {game_agent.name}. Goodbye!")
            break

        if not user_text.strip():
            continue

        # 2. Append User message
        chat_history.append({"role": "user", "content": user_text})

        # 3. Dynamic Re-construction of System Prompt (Refreshes memory changes inside context)
        chat_history[0]["content"] = construct_system_prompt(game_agent, selected_context, conversation_topic)

        # 4. Generate Agent Response
        agent_reply = generate_response(chat_history)
        print(f"{game_agent.name} ({game_agent.role}): {agent_reply}")

        # 5. Commit Reply to conversation tracking
        chat_history.append({"role": "assistant", "content": agent_reply})
        game_agent.update_short_term_memory(f"I replied: {agent_reply}")
        # 6. Automatic Memory Processing (Asynchronous Summary Simulation)
        # Summarizes the round to build out long-term memory milestones
        interaction_summary = summarize_memory(f"User: {user_text} | Agent Reply: {agent_reply}")
        game_agent.update_long_term_memory(interaction_summary)
