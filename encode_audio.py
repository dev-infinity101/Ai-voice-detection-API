import base64
import os
import sys

def encode_audio_to_base64(file_path):
    """
    Reads an audio file and returns its Base64 encoded string.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        with open(file_path, "rb") as audio_file:
            # Read binary data
            binary_data = audio_file.read()
            # Encode to base64
            base64_encoded = base64.b64encode(binary_data)
            # Convert to string
            return base64_encoded.decode('utf-8')
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # Path to the specific sample file
    sample_path = r"d:\PROJECTS FILES\Hackathons\Ai-voice-detection\samples\sample voice 1.mp3"
    
    # Output file to save the base64 string (since it will be very long)
    output_path = r"d:\PROJECTS FILES\Hackathons\Ai-voice-detection\samples\sample-voice-1_base64.txt"

    print(f"Encoding file: {sample_path}")
    base64_string = encode_audio_to_base64(sample_path)

    if base64_string:
        # Save to file
        with open(output_path, "w") as f:
            f.write(base64_string)
        
        print("\n" + "="*50)
        print("SUCCESS!")
        print(f"Base64 string saved to: {output_path}")
        print("="*50)
        print("\nFirst 100 characters of the Base64 string:")
        print(base64_string[:100] + "...")
        print("\nCopy the full content from the text file mentioned above.")
    else:
        print("Failed to encode audio.")

if __name__ == "__main__":
    main()
