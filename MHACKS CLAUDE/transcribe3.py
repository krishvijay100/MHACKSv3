from openai import OpenAI
import json

def generate_feedback(transcript, pauses, eyecontactratio):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a public speaking coach. Focus on clarity, fluency, and engagement. Always respond ONLY with valid JSON."
            },
            {
                "role": "user",
                "content": f"""Analyze the transcript below. 
                Give a score 1-100 for each of the following categories (in this order): Eye contact ratio, hand gesture ratio, fluency (filler words/stutters/pauses), vocabulary. Then give a cumulative score which is an average of the previous factor scores. The answers must be in JSON format as a list under the key 'scores'.
                Pauses: {pauses}, Eye Contact Ratio: {eyecontactratio:.2f} Hand Gesture Ratio: {handgestures.ratio}
                Transcript:
                ```{transcript}```"""
            }
        ],
        response_format = {"type": "json_object"}
    )
    json_text = response.output[0].content[0].text
    data = json.loads(json_text)
    score_eyecontactratio, score_handgestureratio, score_fluency, score_vocab, score_overall = data["scores"]

# add hand gestures to prompt above
