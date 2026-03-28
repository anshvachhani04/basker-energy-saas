# Entry point for Streamlit Cloud root deployment
# This file simply imports and runs the main dashboard
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'streamlit_app'))
exec(open(os.path.join(os.path.dirname(__file__), 'streamlit_app', 'main.py')).read())
