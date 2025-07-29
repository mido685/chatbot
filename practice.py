import edge_tts
import asyncio
import pygame

async def speak(text, voice="en-GB-RyanNeural"):
    filename = "jarvis_response.mp3"
    communicate = edge_tts.Communicate(text, voice=voice, rate='-10%', pitch='-2Hz')
    await communicate.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait until playback finishes
        await asyncio.sleep(0.1)

# Example usage
response = "System online. Awaiting your command."
asyncio.run(speak(response))
