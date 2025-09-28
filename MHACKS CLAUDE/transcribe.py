from openai import OpenAI
import json


def generate_feedback(transcript, pauses, eyecontactratio, handgestures=None):
    """
    Generate feedback for public speaking performance

    Args:
        transcript (str): The speech transcript
        pauses (int): Number of pauses detected
        eyecontactratio (float): Eye contact ratio (0-1)
        handgestures (int, optional): Number of hand gestures detected

    Returns:
        dict: Analysis data with scores
    """
    client = OpenAI()

    # Build the content string with optional hand gestures
    content = f"""Analyze the transcript below. 
    Give a score 1-100 for each of the following categories (in this order): 
    Eye contact ratio, fluency (filler words/stutters/pauses), vocabulary{', hand gestures' if handgestures is not None else ''}. 
    Then give a cumulative score which is an average of the previous factor scores. 
    The answers must be in JSON format as a list under the key 'scores'.

    Pauses: {pauses}, Eye Contact Ratio: {eyecontactratio:.2f}"""

    if handgestures is not None:
        content += f", Hand Gestures: {handgestures}"

    content += f"""
    Transcript:
    ```{transcript}```"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a public speaking coach. Focus on clarity, fluency, and engagement. Always respond ONLY with valid JSON."
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            response_format={"type": "json_object"}
        )

        # Fixed: Access the response content correctly
        json_text = response.choices[0].message.content
        data = json.loads(json_text)

        # Validate that we have the expected scores
        if "scores" not in data:
            raise ValueError("Response does not contain 'scores' key")

        scores = data["scores"]
        expected_length = 4 if handgestures is None else 5

        if len(scores) != expected_length:
            raise ValueError(f"Expected {expected_length} scores, got {len(scores)}")

        # Add metadata to the response
        data["metadata"] = {
            "pauses": pauses,
            "eye_contact_ratio": eyecontactratio,
            "transcript_length": len(transcript),
            "word_count": len(transcript.split()) if transcript else 0
        }

        if handgestures is not None:
            data["metadata"]["hand_gestures"] = handgestures

        return data

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {json_text}")
        # Return default scores in case of error
        return {
            "scores": [50, 50, 50, 50] if handgestures is None else [50, 50, 50, 50, 50],
            "error": "Failed to parse AI response",
            "metadata": {
                "pauses": pauses,
                "eye_contact_ratio": eyecontactratio,
                "transcript_length": len(transcript) if transcript else 0,
                "word_count": len(transcript.split()) if transcript else 0
            }
        }

    except Exception as e:
        print(f"Error generating feedback: {e}")
        # Return default scores in case of error
        return {
            "scores": [50, 50, 50, 50] if handgestures is None else [50, 50, 50, 50, 50],
            "error": str(e),
            "metadata": {
                "pauses": pauses,
                "eye_contact_ratio": eyecontactratio,
                "transcript_length": len(transcript) if transcript else 0,
                "word_count": len(transcript.split()) if transcript else 0
            }
        }


def extract_scores(data):
    """
    Extract individual scores from the analysis data

    Args:
        data (dict): The analysis data returned by generate_feedback

    Returns:
        tuple: Individual scores (eye_contact, fluency, vocabulary, overall, [hand_gestures])
    """
    scores = data.get("scores", [])

    if len(scores) == 4:
        return scores[0], scores[1], scores[2], scores[3]
    elif len(scores) == 5:
        return scores[0], scores[1], scores[2], scores[3], scores[4]
    else:
        # Return default values if scores are malformed
        return 50, 50, 50, 50


# Example usage and testing function
def test_generate_feedback():
    """Test function to verify the feedback generation works"""
    sample_transcript = "Hello everyone, um, today I want to talk about, uh, climate change. It's a very important topic that affects us all."
    sample_pauses = 3
    sample_eye_contact = 0.75
    sample_gestures = 5

    print("Testing feedback generation...")

    # Test without hand gestures
    result1 = generate_feedback(sample_transcript, sample_pauses, sample_eye_contact)
    print("Without hand gestures:")
    print(json.dumps(result1, indent=2))

    # Test with hand gestures
    result2 = generate_feedback(sample_transcript, sample_pauses, sample_eye_contact, sample_gestures)
    print("\nWith hand gestures:")
    print(json.dumps(result2, indent=2))

    return result1, result2


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_generate_feedback()