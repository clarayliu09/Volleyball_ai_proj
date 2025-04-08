import os
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv


class GPTAnalyzer:

    def __init__(self):
        """Initialize the GPT analyzer with API credentials."""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None

    def initialize_client(self) -> bool:
        """
        Initialize the OpenAI client with API key.
        Returns True if successful, False otherwise.
        """
        if not self.api_key:
            return False

        try:
            self.client = OpenAI(api_key=self.api_key)
            return True
        except Exception:
            return False

    def analyze_technique(self, prompt: str) -> Dict:
        """
        Generate a technique analysis using GPT-4.
        
        Args:
            prompt (str): The analysis prompt
            
        Returns:
            dict: Analysis results containing strengths and recommendations
        """
        if not self.client and not self.initialize_client():
            return {
                "error":
                "OpenAI API key not configured. Please add your API key to the .env file or environment variables.",
                "analysis": None
            }

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{
                    "role":
                    "system",
                    "content":
                    "You are an expert volleyball coach specializing in serve analysis. "
                    "Provide clear, specific, and actionable feedback based on the metrics provided."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=1000)

            analysis = response.choices[0].message.content

            return {
                "analysis": analysis,
                "timestamp": os.path.basename(__file__),
                "model": "gpt-4-turbo-preview"
            }

        except Exception as e:
            return {
                "error": f"Error generating analysis: {str(e)}",
                "analysis": None
            }
