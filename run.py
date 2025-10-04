import subprocess
import sys

def main():
    try:
        subprocess.run(["streamlit", "run", "src/app.py"] + sys.argv[1:], check=True)
    except FileNotFoundError:
        print("Error: streamlit is not installed. Please install it with \"pip install streamlit\"")
    except subprocess.CalledProcessError as e:
        print(f"Error running streamlit: {e}")

