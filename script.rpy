# The script of the game goes in this file.

# Declare characters used by this game.
define e = Character("Eileen")
define ug = Character("UkranianGuy")
define b = Character("Jane")

# Image definitions using im.Crop (for sprite sheets)
# Syntax: im.Crop("path/to/image", x, y, width, height)
# Uncomment and adjust coordinates/filenames as needed:
# image eileen happy = im.Crop("images/eileen_spritesheet.png", 0, 0, 400, 600)
# image eileen concerned = im.Crop("images/eileen_spritesheet.png", 400, 0, 400, 600)
# image eileen neutral = im.Crop("images/eileen_spritesheet.png", 800, 0, 400, 600)

# Initialize AI Agent states
default eileen_chat_history = []
default eileen_social_context = "Office environment with open communication and collaboration."
default eileen_topic = "General introduction and corporate assistance."
default eileen_agent = None

default ukranian_guy_chat_history = []
default ukranian_guy_social_context = "Technical department with a focus on problem-solving."
default ukranian_guy_topic = "Technical support and infrastructure."
default ukranian_guy_agent = None

default boss_chat_history = []
default boss_social_context = "Management office with a focus on project oversight and team leadership."
default boss_topic = "Project management and department goals."
default boss_agent = None

init python:
    import sys
    import os
    # Add the agent directory to sys.path to allow imports
    agent_path = os.path.join(config.basedir, "agent")
    if agent_path not in sys.path:
        sys.path.insert(0, agent_path) # Insert at the beginning to prioritize
    
    # Debugging: Log environment info
    with open(os.path.join(config.basedir, "ai_debug_env.log"), "w") as f:
        f.write("config.basedir: " + str(config.basedir) + "\n")
        f.write("sys.path: " + str(sys.path) + "\n")
        f.write("Current Python version: " + str(sys.version) + "\n")
        
        try:
            import transformers
            f.write("transformers found locally\n")
        except ImportError:
            f.write("transformers NOT found locally. Will use AI Bridge Fallback.\n")

    # Import the sentiment analysis function
    try:
        from sentiment_analysis import get_sentiment
    except Exception as err:
        def get_sentiment(text):
            return "NEUTRAL", 0.0

    # Import AI Agent logic
    try:
        from main import Agent, generate_response, construct_system_prompt, summarize_memory, get_agent_from_pool
    except Exception as err:
        import traceback
        ai_error_info = str(err)
        with open(os.path.join(config.basedir, "ai_import_error.log"), "w") as f:
            f.write(ai_error_info + "\n")
            f.write(traceback.format_exc())
        Agent = None
        def generate_response(history, error_msg=ai_error_info): 
            return "I'm sorry, my AI module is offline. Error: " + error_msg
        def construct_system_prompt(a, c, t): return ""
        def get_agent_from_pool(n): return None

# The game starts here.

label start:

    # Show a background. This uses a placeholder by default, but you can
    # add a file (named either "bg room.png" or "bg room.jpg") to the
    # images directory to show it.

    scene bg room

    # This shows a character sprite. A placeholder is used, but you can
    # replace it by adding a file named "eileen happy.png" to the images
    # directory.

    show eileen happy

    # These display lines of dialogue.

    # Initialize Eileen's Agent profile if not already done
    if eileen_agent is None and Agent is not None:
        python:
            eileen_profile = get_agent_from_pool("Eileen")
            if eileen_profile:
                eileen_agent = Agent(
                    name=eileen_profile['name'],
                    role="Corporate Relations Assistant",
                    personality=eileen_profile['personality'],
                    goals=eileen_profile['goals']
                )
            else:
                # Fallback to hardcoded if not in pool
                eileen_agent = Agent(
                    name="Eileen",
                    role="Corporate Relations Assistant",
                    personality="friendly, helpful, positive, and inherently optimistic",
                    goals=["Help the player understand corporate relations", "Maintain a positive atmosphere"]
                )
        
        # Seed initial memories
        $ eileen_agent.update_long_term_memory("Eileen was assigned to the Player's department last quarter.")
        $ eileen_agent.update_long_term_memory("The current corporate goal is to improve team synergy.")
        $ eileen_agent.update_short_term_memory("Eileen just met the Player for a morning briefing.")

    # Initialize UkranianGuy's Agent profile if not already done
    if ukranian_guy_agent is None and Agent is not None:
        python:
            ug_profile = get_agent_from_pool("UkranianGuy")
            if ug_profile:
                ukranian_guy_agent = Agent(
                    name=ug_profile['name'],
                    role="Technical Lead",
                    personality=ug_profile['personality'],
                    goals=ug_profile['goals']
                )
            else:
                # Fallback to hardcoded if not in pool
                ukranian_guy_agent = Agent(
                    name="UkranianGuy",
                    role="Technical Lead",
                    personality="direct, hardworking, resilient, and occasionally humorous",
                    goals=["Ensure technical tasks are completed accurately", "Support the team with practical solutions"]
                )
        
        # Seed initial memories
        $ ukranian_guy_agent.update_long_term_memory("UkranianGuy joined the project as the lead developer.")
        $ ukranian_guy_agent.update_long_term_memory("The infrastructure was recently migrated to a new cloud provider.")
        $ ukranian_guy_agent.update_short_term_memory("UkranianGuy is checking the server logs for any anomalies.")

    # Initialize Boss's Agent profile if not already done
    if boss_agent is None and Agent is not None:
        python:
            boss_profile = get_agent_from_pool("Jane")
            if boss_profile:
                boss_agent = Agent(
                    name=boss_profile['name'],
                    role="Project Manager",
                    personality=boss_profile['personality'],
                    goals=boss_profile['goals']
                )
            else:
                # Fallback to hardcoded if not in pool
                boss_agent = Agent(
                    name="Jane",
                    role="Project Manager",
                    personality="friendly, helpful, positive, and inherently optimistic",
                    goals=["Complete tasks efficiently", "Build strong relationships with colleagues"]
                )
        
        # Seed initial memories
        $ boss_agent.update_long_term_memory("Jane has been managing this department for three years.")
        $ boss_agent.update_short_term_memory("Jane is welcoming the new hire to the team.")

    # Introductory Scene
    "Welcome to your new office. The air is filled with the hum of servers and the quiet clicking of keyboards."
    "It's your first day as a Corporate Relations Liaison."

    show eileen happy at left
    e "Welcome! I'm Eileen, your corporate relations assistant. I'll be helping you navigate the company culture and any administrative needs."

    show ukranianguy neutral at right
    ug "And I'm UkranianGuy. I handle the technical side of things. If the servers go down, it's usually my fault—or I'm the one fixing it."

    show jane neutral at center
    b "Glad to have you on board. I'm Jane, the Project Manager here. We have a lot of work ahead of us, but I'm sure you'll fit right in."

    "The team seems capable and welcoming. You take a moment to settle into your desk."

    hide eileen
    hide ukranianguy
    hide jane

    jump char_selection

label char_selection:
    scene bg room
    menu:
        "Who would you like to talk to?"

        "Talk to Eileen":
            jump eileen_interaction

        "Talk to UkranianGuy":
            jump ukranian_guy_interaction

        "Talk to Jane":
            jump boss_interaction

        "Quit Game":
            return

label eileen_interaction:
    show eileen happy

    e "I'm Eileen, your corporate relations assistant."

    e "How can I help you today? (Type 'exit' to finish our conversation)"

    $ conversation_active = True

    while conversation_active:
        python:
            # Accept text input from the player
            player_input = renpy.input("Your response:", length=100)
            player_input = player_input.strip()

        if not player_input or player_input.lower() in ['exit', 'quit', 'bye']:
            $ conversation_active = False
            e "It was a pleasure talking to you. Have a productive day!"
        else:
            python:
                # 1. Update memory with player input (dialogue is tracked in chat history)
                
                # 2. Prepare/Update chat history
                if not eileen_chat_history:
                    system_p = construct_system_prompt(eileen_agent, eileen_social_context, eileen_topic)
                    eileen_chat_history.append({"role": "system", "content": system_p})
                else:
                    # Refresh system prompt with updated memories
                    eileen_chat_history[0]["content"] = construct_system_prompt(eileen_agent, eileen_social_context, eileen_topic)
                
                eileen_chat_history.append({"role": "user", "content": player_input})
                
                # 3. Generate AI response
                # This might take a few seconds as it's running locally
                ai_reply = generate_response(eileen_chat_history)
                
                # 4. Update history (redundant memory updates removed to prevent repetition)
                eileen_chat_history.append({"role": "assistant", "content": ai_reply})
                
                # 5. Determine Eileen's emotion based on AI reply sentiment
                # This allows Eileen to automatically change expression to match her words
                eileen_sentiment, eileen_score = get_sentiment(ai_reply)
                if eileen_sentiment == "POSITIVE":
                    renpy.show("eileen happy")
                elif eileen_sentiment == "NEGATIVE":
                    renpy.show("eileen concerned")
                else:
                    renpy.show("eileen neutral")
                    
                # 6. Manage history length and consolidate memory
                if len(eileen_chat_history) > 10:
                    # Summarize the oldest exchanges (excluding system prompt at 0)
                    # We'll take 4 messages (2 exchanges) to summarize
                    try:
                        to_summarize = eileen_chat_history[1:5]
                        summary_input = ""
                        for msg in to_summarize:
                            summary_input += f"{msg['role']}: {msg['content']}\n"
                        
                        summary = summarize_memory(summary_input)
                        if eileen_agent:
                            eileen_agent.update_long_term_memory(f"Past interaction summary: {summary}")
                        
                        # Remove the summarized messages from active history
                        del eileen_chat_history[1:5]
                    except:
                        # Fallback: just truncate if summarization fails
                        if len(eileen_chat_history) > 12:
                            del eileen_chat_history[1:5]

            # Eileen speaks the AI-generated response
            e "[ai_reply]"

    hide eileen
    jump char_selection

label ukranian_guy_interaction:
    show ukranianguy neutral

    ug "Hello there. I'm UkranianGuy."

    ug "Is there something technical you need help with? Or just checking in?"

    $ conversation_active = True

    while conversation_active:
        python:
            # Accept text input from the player
            player_input = renpy.input("Your response to UkranianGuy:", length=100)
            player_input = player_input.strip()

        if not player_input or player_input.lower() in ['exit', 'quit', 'bye']:
            $ conversation_active = False
            ug "Understood. I'll get back to the code then. See you around."
        else:
            python:
                # 2. Prepare/Update chat history
                if not ukranian_guy_chat_history:
                    system_p = construct_system_prompt(ukranian_guy_agent, ukranian_guy_social_context, ukranian_guy_topic)
                    ukranian_guy_chat_history.append({"role": "system", "content": system_p})
                else:
                    # Refresh system prompt with updated memories
                    ukranian_guy_chat_history[0]["content"] = construct_system_prompt(ukranian_guy_agent, ukranian_guy_social_context, ukranian_guy_topic)
                
                ukranian_guy_chat_history.append({"role": "user", "content": player_input})
                
                # 3. Generate AI response
                ai_reply = generate_response(ukranian_guy_chat_history)
                
                # 4. Update history
                ukranian_guy_chat_history.append({"role": "assistant", "content": ai_reply})
                
                # 5. Determine UkranianGuy's emotion based on AI reply sentiment
                ug_sentiment, ug_score = get_sentiment(ai_reply)
                if ug_sentiment == "POSITIVE":
                    renpy.show("ukranianguy happy")
                elif ug_sentiment == "NEGATIVE":
                    renpy.show("ukranianguy concerned")
                else:
                    renpy.show("ukranianguy neutral")

                # 6. Manage history length and consolidate memory
                if len(ukranian_guy_chat_history) > 10:
                    try:
                        to_summarize = ukranian_guy_chat_history[1:5]
                        summary_input = ""
                        for msg in to_summarize:
                            summary_input += f"{msg['role']}: {msg['content']}\n"
                        
                        summary = summarize_memory(summary_input)
                        if ukranian_guy_agent:
                            ukranian_guy_agent.update_long_term_memory(f"Past interaction summary: {summary}")
                        
                        del ukranian_guy_chat_history[1:5]
                    except:
                        if len(ukranian_guy_chat_history) > 12:
                            del ukranian_guy_chat_history[1:5]

            # UkranianGuy speaks the AI-generated response
            ug "[ai_reply]"

    hide ukranianguy
    jump char_selection

label boss_interaction:
    show jane neutral

    b "Hello. I'm Jane, the Project Manager. How are things progressing on your end?"

    $ conversation_active = True

    while conversation_active:
        python:
            # Accept text input from the player
            player_input = renpy.input("Your response to Jane:", length=100)
            player_input = player_input.strip()

        if not player_input or player_input.lower() in ['exit', 'quit', 'bye']:
            $ conversation_active = False
            b "Keep up the good work. Let's touch base again soon."
        else:
            python:
                # 2. Prepare/Update chat history
                if not boss_chat_history:
                    system_p = construct_system_prompt(boss_agent, boss_social_context, boss_topic)
                    boss_chat_history.append({"role": "system", "content": system_p})
                else:
                    # Refresh system prompt with updated memories
                    boss_chat_history[0]["content"] = construct_system_prompt(boss_agent, boss_social_context, boss_topic)
                
                boss_chat_history.append({"role": "user", "content": player_input})
                
                # 3. Generate AI response
                ai_reply = generate_response(boss_chat_history)
                
                # 4. Update history
                boss_chat_history.append({"role": "assistant", "content": ai_reply})
                
                # 5. Determine Jane's emotion based on AI reply sentiment
                b_sentiment, b_score = get_sentiment(ai_reply)
                if b_sentiment == "POSITIVE":
                    renpy.show("jane happy")
                elif b_sentiment == "NEGATIVE":
                    renpy.show("jane concerned")
                else:
                    renpy.show("jane neutral")

                # 6. Manage history length and consolidate memory
                if len(boss_chat_history) > 10:
                    try:
                        to_summarize = boss_chat_history[1:5]
                        summary_input = ""
                        for msg in to_summarize:
                            summary_input += f"{msg['role']}: {msg['content']}\n"
                        
                        summary = summarize_memory(summary_input)
                        if boss_agent:
                            boss_agent.update_long_term_memory(f"Past interaction summary: {summary}")
                        
                        del boss_chat_history[1:5]
                    except:
                        if len(boss_chat_history) > 12:
                            del boss_chat_history[1:5]

            # Jane speaks the AI-generated response
            b "[ai_reply]"

    hide jane
    jump char_selection
