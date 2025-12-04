## This is for an interface class for diarization models
from typing import List, Dict
class Diarizer:
    def diarize(self, audio_path) -> List[Dict]:
        """
        Diarize the given audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:

            List[Dict]: A list of dictionaries containing speaker segments.
            [
                {
                    "start" : float,  # Start time of the segment in seconds
                    "end"   : float,  # End time of the segment in seconds  
                    "speaker_id" : str # Identifier for the speaker
                    ## Text Emotion , Language to be added later
                }
            ]
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
        ## Its like throwing an error if the method is not implemented in the subclass

## This is required since we will be using multiple diarization models -> 
# Each model will inherit from this base class and implement the diarize method

## pyannote diarization model subclass
## CustomKmeansDiarization model will be another subclass

## Example return format:
"""
[
  {
    "start": 0.5,
    "end": 3.2,
    "speaker": "Speaker 1"
  },
  {
    "start": 3.2,
    "end": 5.7,
    "speaker": "Speaker 2"
  }
]
"""