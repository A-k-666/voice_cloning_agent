"""
OpenAI Client for Filler Messages and Small Talk
"""

import os
import random
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv(".env.local")

logger = logging.getLogger("openai-client")

API_KEY = os.getenv("OPENAI_API_KEY")

# Fun facts for filler messages during voice cloning
FUN_FACTS = [
    "Did you know honey never spoils?",
    "Bananas are berries, but strawberries aren't!",
    "Octopuses have three hearts.",
    "A group of flamingos is called a flamboyance.",
    "Wombat poop is cube-shaped!",
    "Your cloned voice will sound just like you!",
    "Sharks have been around longer than trees.",
    "A day on Venus is longer than its year.",
    "There are more possible chess games than atoms in the universe.",
    "A single cloud can weigh more than a million pounds.",
]


class OpenAIClient:
    """Client for OpenAI filler messages"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.facts = FUN_FACTS.copy()
    
    async def random_fact(self) -> str:
        """Get a random fun fact"""
        fact = random.choice(self.facts)
        logger.debug(f"🎲 Random fact: {fact}")
        return fact
    
    async def get_filler_message(self) -> str:
        """Get a filler message (alias for random_fact)"""
        return await self.random_fact()





