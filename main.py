import contextlib
from memory import Memory
from classifier_pro import IntentClassifierONNX  # ✅ import your ONNX classifier
from func_equip import EquipmentQuery
from extraction import extract_entities
import json
import random
import sys
import os  
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1" 
import asyncio
import edge_tts
import pygame
import psutil
import getpass
from datetime import datetime
memory = Memory()
# Load response templates
with open(r"C:\Users\AL_hamd\OneDrive\Desktop\Chatbot\Config\responses.json", "r", encoding="utf-8") as f:
    RESPONSE_TEMPLATES = json.load(f)

# Load your CSV data
query_tool = EquipmentQuery("equipments.csv")


# Load intent classifier
intent_clf = IntentClassifierONNX(
    'models/logistic_classifier.joblib',
    'models/label_encoder2.joblib',
    r'C:\Users\AL_hamd\OneDrive\Desktop\Chatbot\onnx_model\model.onnx',
    'sentence-transformers/paraphrase-MiniLM-L3-v2'
)
async def startup_message():
    username = getpass.getuser()
    now = datetime.now().strftime("%I:%M %p")

    await speak(f"Good { 'morning' if datetime.now().hour < 12 else 'afternoon' }, {username}.")
    await speak(f"The time is {now}.")
    await speak("Initializing JARVIS protocol.")
    await asyncio.sleep(1)
    await speak("Loading core modules.")
    await asyncio.sleep(1)
    await speak("Neural network connections stable.")
    await asyncio.sleep(0.5)
    await speak("Establishing voice interface.")
    await asyncio.sleep(0.5)
    await speak("Activating system diagnostics.")
    await report_system_status()
    await speak("All functionalities are operational. Standing by.")



async def report_system_status():
    battery = psutil.sensors_battery()
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()

    status = (
        f"System check complete. "
        f"Battery at {battery.percent} percent. "
        f"CPU usage at {cpu} percent. "
        f"Memory usage at {ram.percent} percent."
    )
    await speak(status)

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

async def speak(text, voice="en-GB-RyanNeural", rate='-10%', pitch='-2Hz'):
    filename = "jarvis_response.mp3"

    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
        pygame.mixer.quit()

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(filename)

    with suppress_stdout():
        pygame.mixer.init()

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)
def generate_response(user_input,intent,confidence):
    (intent, confidence) = intent_clf.predict_intent(user_input)
    entities = extract_entities(user_input)
    equipment_code = entities.get("equipment") or memory.get("equipment")
    branch = entities.get("branch") or memory.get("branch")
    if confidence < 0.7:
        return "🤔 I'm not sure what you mean. Could you rephrase?"
    else:
        memory.update(intent=intent, entities=entities)
    if intent == "ask_the_equipment_in_branch":
        if equipment_code and branch:
            result = query_tool.get_quantity_of_branch(equipment_code, branch)
            if not result["success"]:
                return result["message"]
            return random.choice(RESPONSE_TEMPLATES["get_quantity"]).format(
                equipment=result["equipment"],
                branch=result["branch"],
                count=result["quantity"]
            )
        elif not equipment_code:
            return "Which equipment are you referring to?"
        elif not branch:
                return "Which branch do you mean?"
    elif intent == "ask_equipment_total_only":
        if equipment_code:
            result = query_tool.get_total_equipment_only(equipment_code)
            if "No data" in result:
                return random.choice(RESPONSE_TEMPLATES["get_total_equipment_only_not_found"])
            count = result.spxlit(":")[-1].strip()
            return random.choice(RESPONSE_TEMPLATES["get_total_equipment_only"]).format(
                equipment=equipment_code, count=count
            )
        return "Which equipment do you want to check?"

    elif intent == "greeting":
        return random.choice(RESPONSE_TEMPLATES["greeting"])

    elif intent == "farewell":
        return random.choice(RESPONSE_TEMPLATES["farewell"])
    
    

    return "I understood your message, but I'm not trained to handle that yet."
async def chat_loop():
    await startup_message() 
    while True:
        user_input = input("You: ")
        (intent, confidence) = intent_clf.predict_intent(user_input)
        if intent == "farewell" and confidence >= 0.7:
            print("Bot:", random.choice(RESPONSE_TEMPLATES["farewell"]))
            await speak("goodbye")
            break
        response = generate_response(user_input,intent,confidence)
        print("Bot:", response)
        await speak(response)

# ✅ Entry point
if __name__ == "__main__":
    asyncio.run(chat_loop())
