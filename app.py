"""
Gemini GenAI Trivia Challenge
A trivia game powered by Google's Gemini AI model that:
- Generates unique questions based on any topic
- Tracks scores and maintains leaderboards
- Provides timed questions with point multipliers
- Saves game history to Google Sheets
"""

import streamlit as st
import google.generativeai as genai
import time
import google.ai.generativelanguage as glm
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import os
import json
import random

import atexit
import grpc
import logging

os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GRPC_DNS_RESOLVER'] = 'native'
logging.getLogger('absl').setLevel(logging.ERROR)


# Initialize GEMINI client using environment variable or Streamlit secrets
def get_gemini_key():
    """Get Gemini API key from environment or Streamlit secrets"""
    if os.getenv("GEMINI_API_KEY"):  # Local development
        return os.getenv("GEMINI_API_KEY")
    else:  # Streamlit Cloud
        return st.secrets["gemini"]["GEMINI_API_KEY"]

# Initialize Gemini
genai.configure(api_key=get_gemini_key())
model = genai.GenerativeModel('gemini-1.5-flash')



# Set page configuration
st.set_page_config(
    page_title="🧠 Gemini GenAI Trivia Challenge",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide Streamlit's default header, footer, and menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom CSS for styling

st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');
    
    /* [All your existing styles remain the same until the footer social section] */
    
    .main {
        padding: 0.5rem;
    }
    
    /* [Keep all your existing styles...] */
    
    /* Updated Footer Social Section */
    .footer-social {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 20px 0;
    }

    .social-button {
        padding: 8px 15px;
        border-radius: 20px;
        text-decoration: none;
        color: #333;  /* Darker text color */
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        transition: all 0.3s ease;
        font-weight: 500;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .social-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    .linkedin-button { 
        background-color: rgba(0, 119, 181, 0.15);
        color: #0077B5;
    }
    .linkedin-button:hover { 
        background-color: rgba(0, 119, 181, 0.25);
    }

    .facebook-button { 
        background-color: rgba(24, 119, 242, 0.15);
        color: #1877F2;
    }
    .facebook-button:hover { 
        background-color: rgba(24, 119, 242, 0.25);
    }

    .twitter-button { 
        background-color: rgba(29, 161, 242, 0.15);
        color: #1DA1F2;
    }
    .twitter-button:hover { 
        background-color: rgba(29, 161, 242, 0.25);
    }

    .email-button { 
        background-color: rgba(234, 67, 53, 0.15);
        color: #EA4335;
    }
    .email-button:hover { 
        background-color: rgba(234, 67, 53, 0.25);
    }

    .social-button i {
        font-size: 1.2rem;
        margin-right: 4px;
    }
    
    .timer-container {
        margin: 10px 0;
        padding: 10px;
        border-radius: 8px;
        background: #f0f2f6;
    }

    .timer-warning {
        background: rgba(255, 87, 51, 0.1);
    }

    .progress-bar {
        width: 100%;
        height: 8px;
        background: #ddd;
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-bar-fill {
        height: 100%;
        background: #1E88E5;
        transition: width 0.5s ease;
    }

    .question-display {
        margin: 20px 0;
        padding: 15px;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* [Keep rest of your existing styles] */
    
    </style>
""", unsafe_allow_html=True)

# Initialize all required session state variables
def initialize_session_state():
    """Initialize all required session state variables for the trivia game"""
    
    # Player and Game State
    if 'player_name' not in st.session_state:
        st.session_state.player_name = ""
    if 'topic' not in st.session_state:
        st.session_state.topic = ""
    if 'questions_asked' not in st.session_state:
        st.session_state.questions_asked = 0
    if 'total_score' not in st.session_state:
        st.session_state.total_score = 0
    if 'game_length' not in st.session_state:
        st.session_state.game_length = 10
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
        
    # Question Management
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'answer_selected' not in st.session_state:
        st.session_state.answer_selected = False
    if 'feedback' not in st.session_state:
        st.session_state.feedback = None
    if 'question_cache' not in st.session_state:
        st.session_state.question_cache = set()
        
    # Question History and Tracking
    if 'question_history' not in st.session_state:
        st.session_state.question_history = {
            'questions': [],    # Track question metadata
            'concepts': set(),  # Track used concepts
            'domains_used': [] # Track used knowledge domains
        }
        
    # Leaderboard Management
    if 'leaderboard_cache' not in st.session_state:
        st.session_state.leaderboard_cache = None
    if 'last_sheet_load' not in st.session_state:
        st.session_state.last_sheet_load = None
    if 'sheet_object' not in st.session_state:
        st.session_state.sheet_object = None

# Call this at the start of your app

    
    

# Update reset_game_state function to handle question history

def reset_game_state():
    """Reset game state for a new game"""
    st.session_state.questions_asked = 0
    st.session_state.total_score = 0
    st.session_state.current_question = None
    st.session_state.answer_selected = False
    st.session_state.feedback = None
    st.session_state.game_active = False
    
    # Clear question history but maintain structure
    st.session_state.question_history = {
        'questions': [],
        'concepts': set(),
        'domains_used': []
    }
    
    # Clear caches
    st.session_state.question_cache = set()
    st.session_state.leaderboard_cache = None
    st.session_state.last_sheet_load = None

    

def load_and_resize_image(image_path, width=None, max_size=None):
    """Load an image and resize it with maximum dimensions"""
    try:
        from PIL import Image
        image = Image.open(image_path)
        
        if max_size:
            # Calculate ratio to maintain aspect ratio
            ratio = min(max_size[0]/float(image.size[0]), 
                       max_size[1]/float(image.size[1]))
            new_size = (int(float(image.size[0])*float(ratio)), 
                       int(float(image.size[1])*float(ratio)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            return image
        elif width:
            ratio = width/float(image.size[0])
            height = int(float(image.size[1])*float(ratio))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            return image
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None




def is_running_on_streamlit():
    """Check if the code is running on Streamlit Cloud"""
    checks = [
        hasattr(st, "secrets"),
        any(key.startswith("STREAMLIT_") for key in os.environ),
        not os.path.exists("new-year-trivia-game-932d8241aa4e.json")
    ]
    return any(checks)



def authenticate_google_sheets():
    """Authenticate with Google Sheets API and return sheet object"""
    # Check if we have a cached sheet object
    if st.session_state.sheet_object is not None:
        return st.session_state.sheet_object
    
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    
    try:
        if is_running_on_streamlit():
            creds_dict = dict(st.secrets["gcp_service_account"])
            
            if "private_key" in creds_dict:
                pk = creds_dict["private_key"]
                if isinstance(pk, str):
                    pk = pk.replace('\\n', '\n').strip()
                    if not pk.endswith('\n'):
                        pk += '\n'
                    creds_dict["private_key"] = pk
            
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            
        else:
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                "new-year-trivia-game-932d8241aa4e.json", 
                scope
            )
            
        client = gspread.authorize(creds)
        sheet_url = (st.secrets["google_sheets"]["url"] 
                    if is_running_on_streamlit() 
                    else "https://docs.google.com/spreadsheets/d/1vs_JYu7HqmGiVUZjTdiDemVBhj3APV90Z5aa1jt56-g/edit#gid=0")
        
        spreadsheet = client.open_by_url(sheet_url)
        # Cache the sheet object
        st.session_state.sheet_object = spreadsheet.sheet1
        return st.session_state.sheet_object
        
    except Exception as e:
        return None
    

def load_leaderboard(sheet, force_refresh=False):
    """Load leaderboard data with caching"""
    from datetime import datetime, timedelta
    
    # Check if we have cached data and it's less than 5 minutes old
    if (not force_refresh and 
        st.session_state.leaderboard_cache is not None and 
        st.session_state.last_sheet_load is not None and 
        datetime.now() - st.session_state.last_sheet_load < timedelta(minutes=5)):
        return st.session_state.leaderboard_cache
    
    if sheet is None:
        return {}
        
    try:
        # Load from sheet
        data = sheet.get_all_records()
        leaderboard = {i: {
            "name": row["Name"],
            "score": row["Score"],
            "topic": row["Topic"],
            "date": row["Date"],
            "time": row["Time"],
            "questions_answered": row["Questions_Answered"],
            "game_length": row["Game_Length"]
        } for i, row in enumerate(data)}
        
        # Update cache
        st.session_state.leaderboard_cache = leaderboard
        st.session_state.last_sheet_load = datetime.now()
        
        return leaderboard
    except Exception as e:
        print(f"Error loading leaderboard: {str(e)}")
        return {}


def save_leaderboard(sheet, leaderboard):
    """Save leaderboard data to Google Sheet with extended fields and rate limiting"""
    if sheet is None:
        return
    
    import time
    from random import uniform
    
    max_retries = 5
    base_wait = 1  # Base wait time in seconds
    
    for attempt in range(max_retries):
        try:
            # Clear the sheet and set headers
            sheet.clear()
            time.sleep(uniform(1, 2))  # Random delay between 1-2 seconds
            
            headers = ["Name", "Score", "Topic", "Date", "Time", 
                      "Questions_Answered", "Game_Length"]
            sheet.append_row(headers)
            time.sleep(uniform(1, 2))  # Random delay between 1-2 seconds
            
            # Sort entries by date/time (newest first) and score (highest first)
            sorted_entries = sorted(
                leaderboard.values(),
                key=lambda x: (x["date"], x["time"], -x["score"]),
                reverse=True
            )
            
            # Append entries with rate limiting
            for entry in sorted_entries:
                # Add random delay between writes
                time.sleep(uniform(0.5, 1))  # Random delay between 0.5-1 seconds
                
                sheet.append_row([
                    entry["name"],
                    entry["score"],
                    entry["topic"],
                    entry["date"],
                    entry["time"],
                    entry["questions_answered"],
                    entry["game_length"]
                ])
            
            return  # Success
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error saving leaderboard after {max_retries} attempts: {str(e)}")
                return
            
            # Calculate wait time with exponential backoff
            wait_time = (2 ** attempt) * base_wait + uniform(0, 1)
            print(f"Attempt {attempt + 1} failed, waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
 

def update_leaderboard_entry(sheet, player_name, score, topic, questions_answered, game_length, force_write=False):
    """Update leaderboard with local caching"""
    try:
        from datetime import datetime
        current_time = datetime.now()
        
        # Format date and time
        date_str = current_time.strftime("%b %d, %Y")
        time_str = current_time.strftime("%I:%M %p")
        
        # Get cached leaderboard or load if needed
        leaderboard = st.session_state.leaderboard_cache
        if leaderboard is None:
            leaderboard = load_leaderboard(sheet)
        
        # Check for duplicate
        duplicate_exists = any(
            entry["name"] == player_name and
            entry["topic"] == topic and
            entry["score"] == score and
            entry["date"] == date_str
            for entry in leaderboard.values()
        )
        
        if not duplicate_exists:
            # Create new entry
            new_entry = {
                "name": player_name,
                "score": score,
                "topic": topic,
                "date": date_str,
                "time": time_str,
                "questions_answered": questions_answered,
                "game_length": game_length
            }
            
            # Add to local cache
            next_index = max(leaderboard.keys(), default=-1) + 1
            leaderboard[next_index] = new_entry
            st.session_state.leaderboard_cache = leaderboard
            
            # Only write to sheet if forced (end of game)
            if force_write and sheet:
                save_leaderboard_efficient(sheet, leaderboard)
            
            return True
        return False
        
    except Exception as e:
        print(f"Error updating leaderboard: {str(e)}")
        return False

def save_leaderboard_efficient(sheet, leaderboard):
    """Efficient single-operation save to Google Sheet"""
    if sheet is None:
        return
    
    try:
        # Prepare all data at once
        headers = ["Name", "Score", "Topic", "Date", "Time", 
                  "Questions_Answered", "Game_Length"]
        
        # Sort entries
        sorted_entries = sorted(
            leaderboard.values(),
            key=lambda x: (x["date"], x["time"], -x["score"]),
            reverse=True
        )
        
        # Create all rows at once
        rows = [headers] + [
            [
                entry["name"],
                entry["score"],
                entry["topic"],
                entry["date"],
                entry["time"],
                entry["questions_answered"],
                entry["game_length"]
            ]
            for entry in sorted_entries
        ]
        
        # Single batch update operation - FIXED order of arguments
        sheet.update(values=rows, range_name='A1', value_input_option='RAW')
        
    except Exception as e:
        print(f"Error saving leaderboard: {str(e)}")

def get_topic_rankings(leaderboard, topic):
    """Get rankings for a specific topic, sorted by score and date"""
    topic_scores = [
        (entry["name"], entry["score"], entry["date"], entry["time"])
        for entry in leaderboard.values()
        if entry["topic"].lower() == topic.lower()
    ]
    # Sort by score (descending) and then by date/time (newest first)
    return sorted(
        topic_scores,
        key=lambda x: (-x[1], x[2], x[3]),
        reverse=False
    )


def display_game_over(player_name, score, topic, questions_answered, game_length):
    """Enhanced game over display with updated rankings"""
    try:
        sheet = authenticate_google_sheets()
        if sheet:
            # Add these lines right here, before update_leaderboard_entry
            from datetime import datetime
            current_time = datetime.now()
            current_date = current_time.strftime("%b %d, %Y")
            current_time_str = current_time.strftime("%I:%M %p")
            
            # Update leaderboard with force_write=True to save to sheet
            update_leaderboard_entry(
                sheet, player_name, score, topic, 
                questions_answered, game_length,
                force_write=True
            )
            
            # Use cached data for display
            leaderboard = st.session_state.leaderboard_cache
            if leaderboard is None:
                leaderboard = load_leaderboard(sheet)    
            
            

        
            # Calculate rankings
            sorted_entries = sorted(
                leaderboard.values(),
                key=lambda x: (-x["score"], x["date"], x["time"])
            )
            
            overall_rank = next(
                (i + 1 for i, entry in enumerate(sorted_entries)
                if entry["name"] == player_name and 
                entry["score"] == score and
                entry["topic"] == topic),
                None
            )
            
            topic_rankings = get_topic_rankings(leaderboard, topic)
            topic_rank = next(
                (i + 1 for i, (name, s, _, _) in enumerate(topic_rankings)
                if name == player_name and s == score),
                None
            )
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="game-over-section">
                    <div class="game-over-title">🎉 Game Over!</div>
                    <div class="game-over-stats">
                        Final Score: {score}<br>
                        Player: {player_name}<br>
                        Topic: {topic}<br>
                        Questions: {questions_answered}/{game_length}<br>
                        Overall Rank: #{overall_rank}<br>
                        Topic Rank: #{topic_rank}<br>
                        Date: {current_date}<br>
                        Time: {current_time_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🏆 Overall Top 5")
                for i, entry in enumerate(sorted_entries[:5], 1):
                    if (entry["name"] == player_name and 
                        entry["score"] == score and 
                        entry["topic"] == topic):
                        st.markdown(
                            f"""**{i}. {entry['name']} ({entry['topic']}): """
                            f"""{entry['score']} points** ← You"""
                            f""" ({entry['date']} {entry['time']})"""
                        )
                    else:
                        st.markdown(
                            f"""{i}. {entry['name']} ({entry['topic']}): """
                            f"""{entry['score']} points"""
                            f""" ({entry['date']} {entry['time']})"""
                        )
            
            with col2:
                st.markdown(f"### 🎯 Top 5 for {topic}")
                for i, (name, score, date, time) in enumerate(topic_rankings[:5], 1):
                    if name == player_name:
                        st.markdown(
                            f"""**{i}. {name}: {score} points** ← You"""
                            f""" ({date} {time})"""
                        )
                    else:
                        st.markdown(
                            f"""{i}. {name}: {score} points"""
                            f""" ({date} {time})"""
                        )
                
                if topic_rank and topic_rank <= 5:
                    st.success(
                        f"🌟 Congratulations! "
                        f"You are #{topic_rank} on the {topic} leaderboard!"
                    )
            
    except Exception as e:
        st.error(f"Unable to update leaderboard: {str(e)}")

def display_leaderboards():
    """Display both overall and topic-specific leaderboards with timestamps"""
    st.sidebar.markdown("---")
    
    # Get fresh data once for both leaderboards
    sheet = authenticate_google_sheets()
    current_leaderboard = load_leaderboard(sheet) if sheet else {}
    
    # Overall Leaderboard
    with st.sidebar.expander("📊 Overall Leaderboard", expanded=False):
        if current_leaderboard:
            sorted_entries = sorted(
                current_leaderboard.values(),
                key=lambda x: (-x["score"], x["date"], x["time"])
            )
            
            for i, entry in enumerate(sorted_entries[:10], 1):
                st.write(
                    f"""{i}. {entry['name']} ({entry['topic']}): """
                    f"""{entry['score']} points"""
                    f""" - {entry['date']} {entry['time']}"""
                )
        else:
            st.info("Leaderboard temporarily unavailable")
    
    # Topic Leaderboard
    if st.session_state.topic:
        with st.sidebar.expander(
            f"🎯 {st.session_state.topic} Leaderboard", 
            expanded=False
        ):
            if current_leaderboard:
                topic_rankings = get_topic_rankings(current_leaderboard, st.session_state.topic)
                if topic_rankings:
                    for i, (name, score, date, time) in enumerate(topic_rankings[:10], 1):
                        st.write(
                            f"{i}. {name}: {score} points "
                            f"- {date} {time}"
                        )
                else:
                    st.info("No scores yet for this topic!")
            else:
                st.info("Topic leaderboard temporarily unavailable")



def generate_trivia_question(topic):
    """Generate engaging, mentor-style trivia questions that build knowledge progressively"""
    
    # Initialize question difficulty based on questions asked
    question_number = st.session_state.questions_asked + 1
    game_length = st.session_state.game_length
    position = question_number / game_length
    
    # Determine question type
    if position <= 0.2:
        difficulty = "Insightful"
    elif position <= 0.4:
        difficulty = "Fun"
    elif position <= 0.6:
        difficulty = "Practical"
    elif position <= 0.8:
        difficulty = "Educational"
    else:
        difficulty = "Advanced"
    
    # Knowledge domains
    knowledge_domains = [
        "Key Concepts",
        "Example Problems",
        "Surprising Facts",
        "Examples of Use",
        "Economic Value?",
        "Why this Matters",
        "Famous People",
        "Myths and Misconceptions",
        "History, Names, Events, People and Places",
        "Scientific Principles, Formulas, and Experiments",
        "Future Trends"
    ]
    
    # Domain selection
    used_domains = [q.get('domain') for q in st.session_state.question_history.get('questions', [])]
    available_domains = [d for d in knowledge_domains if d not in used_domains[-3:]]
    selected_domain = random.choice(available_domains if available_domains else knowledge_domains)

    # Construct prompt
    prompt = f"""As an expert TRIVIA GAME SHOW HOST and and expert mentor, create ACCURATE, FACT-CHECKED fun and engaging {difficulty}-level question about {topic} 
    focusing on example questions related to the topic requested by the {selected_domain} that people would LOVE to know about. Use Examples.  
    

    FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

    QUESTION: [One clear, concise question (1 sentence)]
    A) [Brief answer]
    B) [Brief answer]
    C) [Brief answer]
    D) [Brief answer]
    CORRECT: [Just the letter: A, B, C, or D]
    FACT_CHECK: [1-2 sentences explaining the correct answer with details]
    KEY_CONCEPTS: [4-6 key terms defined if used in the question or answers]
    DIFFICULTY: {difficulty}
    DOMAIN: {selected_domain}

    REQUIREMENTS:
    - Question must be ONE or TWO CONCISE sentences and interesting and engaging
    - Each answer choice must be ONE or TWO CONCISE SHORT sentences
    - Focus on fascinating relevant information and insights people would want to know about
    - Make it accessible and interesting to a broad general audience (not just experts)
    - Include practical or surprising information
    """

    # Generation configuration
    generation_config = genai.types.GenerationConfig(
        temperature=1.2,
        top_p=0.7,
        top_k=40,
        frequency_penalty=1.0,
        presence_penalty=1.0,
        max_output_tokens=1024
    )

    # Safety settings
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]

    def parse_response(response_text):
        """Parse the response into required format"""
        try:
            lines = response_text.strip().split('\n')
            question = None
            choices = []
            correct = None
            fact_check = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('QUESTION:'):
                    question = line.replace('QUESTION:', '').strip()
                elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                    choices.append(line)
                elif line.startswith('CORRECT:'):
                    correct = line.replace('CORRECT:', '').strip()
                elif line.startswith('FACT_CHECK:'):
                    fact_check = line.replace('FACT_CHECK:', '').strip()
            
            if not all([
                question,
                len(choices) == 4,
                correct in ['A', 'B', 'C', 'D'],
                fact_check
            ]):
                return None

            return {
                "question": question,
                "choices": choices,
                "correct": correct,
                "fact_check": fact_check,
                "start_time": time.time()
            }
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return None

    # Question generation loop
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            if not hasattr(response, 'text') or not response.text:
                continue

            # Parse response
            question_data = parse_response(response.text)
            if not question_data:
                if attempt == max_attempts - 1:
                    st.error("Unable to generate a valid question")
                continue

            # Create unique key for question
            question_key = f"{topic}:{question_data['question']}"
            
            # Check uniqueness
            if question_key in st.session_state.question_cache:
                if attempt == max_attempts - 1:
                    st.error("Unable to generate a unique question")
                continue
            
            # Add to cache
            st.session_state.question_cache.add(question_key)
            
            # Update question history
            if 'questions' not in st.session_state.question_history:
                st.session_state.question_history['questions'] = []
            
            st.session_state.question_history['questions'].append({
                'domain': selected_domain,
                'difficulty': difficulty
            })

            return question_data

        except Exception as e:
            if attempt == max_attempts - 1:
                st.error(f"Error generating question: {str(e)}")
                return None
            time.sleep(2 ** attempt)

    return None



def calculate_score(time_remaining):
    """Calculate score based on remaining time"""
    max_time = 60
    if time_remaining <= 0:
        return 0
    score = int((time_remaining / max_time) * 200)
    return min(200, max(0, score))

def check_answer(selected_answer):
    """Check if the answer is correct and calculate score"""
    current_time = time.time()
    time_elapsed = current_time - st.session_state.current_question["start_time"]
    time_remaining = max(0, 65 - time_elapsed)
    
    correct = selected_answer == st.session_state.current_question["correct"]
    points = calculate_score(time_remaining) if correct else 0
    
    if correct:
        st.session_state.total_score += points
        st.session_state.feedback = (
            f"""✨ Correct! You earned {points} points!
            
            {st.session_state.current_question['fact_check']}""",
            "success"
        )
    else:
        st.session_state.feedback = (
            f"""❌ Wrong! The correct answer was {st.session_state.current_question['correct']}.
            
            {st.session_state.current_question['fact_check']}""",
            "error"
        )
    
    st.session_state.answer_selected = True
    return time_remaining
    
def main():
    """Main function for the Gemini GenAI Trivia Challenge"""
    
    # Initialize Gemini and check availability
    try:
        genai.configure(api_key=get_gemini_key())
        # Quick model check
        test_response = model.generate_content("Test connection")
        if not test_response.text:
            raise Exception("Unable to connect to Gemini model")
    except Exception as e:
        st.error(f"Failed to initialize Gemini API. Please check your API key configuration. Error: {str(e)}")
        st.stop()
        
    # Add auto-refresh script
    st.markdown("""
        <script>
            function refreshPage() {
                if (!document.hidden) {
                    window.location.reload();
                }
            }
            setInterval(refreshPage, 1000);
        </script>
    """, unsafe_allow_html=True)
    
    try:
        # Main title section
        col1, col2 = st.columns([9, 1])
        with col1:
            st.markdown('<h1 class="game-title">🧠 Gemini GenAI Trivia Challenge 🌟</h1>', unsafe_allow_html=True)
        with col2:
            try:
                logo = load_and_resize_image("AppImage.png", max_size=(295, 295))
                if logo:
                    st.image(logo, use_container_width=True)
            except Exception as e:
                st.warning("Unable to load app image")

        # Simplified sidebar with instructions
        st.sidebar.markdown('<div class="instruction-text">', unsafe_allow_html=True)
        st.sidebar.markdown("""
        # 🏆 Game Quick Start: 
        ### 1. 👤 Enter your name & custom Trivia topic, 2. 🔄 Update BOTH using buttons, 3. 🎲 Set questions (5 or 10), 4. 🚀 Press Start Game! to Play, 5. At the End of Game, Press End Game to save your score on leaderboard. 🎯
        """)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

        # Game Controls section
        st.sidebar.markdown('<div class="game-controls">', unsafe_allow_html=True)
        st.sidebar.markdown("# 🎮  Game Controls:")

        # Game length selection
        game_length = st.sidebar.radio(
            " ",
            options=["5 Questions", "10 Questions"],
            index=1 if st.session_state.game_length == 10 else 0,
            horizontal=True,
            label_visibility="collapsed"
        )
        st.session_state.game_length = 10 if "10" in game_length else 5
        
        # Player Input Controls
        with st.sidebar.container():
            st.markdown("# Player Settings")
            new_name = st.text_input("Player Name:", 
                                   value=st.session_state.player_name,
                                   key="player_name_input",
                                   help="Required to start the game")
            new_topic = st.text_input("Trivia Topic:", 
                                    value=st.session_state.topic,
                                    key="topic_input",
                                    help="Required to start the game")
            
            # Update buttons in two columns
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("Update Name", use_container_width=True):
                    if new_name.strip():
                        st.session_state.player_name = new_name.strip()
                        st.rerun()
                    else:
                        st.sidebar.error("Please enter a player name")
            
            with col2:
                if st.button("Update Topic", use_container_width=True):
                    if new_topic.strip():
                        st.session_state.topic = new_topic.strip()
                        st.session_state.current_question = None
                        st.rerun()
                    else:
                        st.sidebar.error("Please enter a topic")

        # Game Control Buttons
        if not st.session_state.game_active:
            if st.sidebar.button("Start Game", use_container_width=True, type="primary"):
                if st.session_state.player_name.strip() and st.session_state.topic.strip():
                    st.session_state.game_active = True
                    reset_game_state()
                    st.session_state.game_active = True
                    st.rerun()
                else:
                    st.sidebar.error("Please enter both player name and topic to start!")
        else:
            if st.sidebar.button("End Game", use_container_width=True, type="secondary"):
                sheet = authenticate_google_sheets()
                if st.session_state.questions_asked > 0:
                    update_leaderboard_entry(
                        sheet,
                        st.session_state.player_name,
                        st.session_state.total_score,
                        st.session_state.topic,
                        st.session_state.questions_asked,
                        st.session_state.game_length,
                        force_write=True
                    )
                reset_game_state()
                st.rerun()
            
            if st.sidebar.button("Start New Game", use_container_width=True, type="primary"):
                sheet = authenticate_google_sheets()
                if st.session_state.questions_asked > 0:
                    update_leaderboard_entry(
                        sheet,
                        st.session_state.player_name,
                        st.session_state.total_score,
                        st.session_state.topic,
                        st.session_state.questions_asked,
                        st.session_state.game_length,
                        force_write=True
                    )
                reset_game_state()
                st.rerun()

        if st.sidebar.button("Reset All", use_container_width=True, type="secondary"):
            st.session_state.question_cache.clear()
            reset_game_state()
            st.rerun()

        # Player Stats
        if st.session_state.player_name:
            st.sidebar.markdown(f"""
            <div class="player-info">
            👤 Player: {st.session_state.player_name}<br>
            💫 Total Score: {st.session_state.total_score}<br>
            📝 Questions: {st.session_state.questions_asked}/{st.session_state.game_length}<br>
            🎯 Topic: {st.session_state.topic}
            </div>
            """, unsafe_allow_html=True)
        
        # Display leaderboards
        display_leaderboards()

        # Main Game Area
        if st.session_state.game_active and st.session_state.questions_asked < st.session_state.game_length:
            if not st.session_state.current_question:
                with st.spinner("Loading next question..."):
                    st.session_state.current_question = generate_trivia_question(st.session_state.topic)
                    st.session_state.answer_selected = False
                    st.session_state.feedback = None
            
            if st.session_state.current_question:
                # Timer and current stats
                current_time = time.time()
                time_elapsed = current_time - st.session_state.current_question["start_time"]
                time_remaining = max(0, 65 - time_elapsed)
                
                timer_class = "timer-warning" if time_remaining < 10 else ""
                progress_percentage = (time_remaining / 65) * 100

                st.markdown(f"""
                    <div class="timer-container {timer_class}">
                        <div class="timer-display">
                            ⏱️ {int(time_remaining)}s
                        </div>
                        <div class="progress-bar">
                            <div class="progress-bar-fill" style="width: {progress_percentage}%;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Auto-submit when time runs out
                if time_remaining <= 0 and not st.session_state.answer_selected:
                    st.session_state.answer_selected = True
                    st.session_state.feedback = (
                        f"""⏰ Time's up! The correct answer was {st.session_state.current_question['correct']}.
                        
                        {st.session_state.current_question['fact_check']}""",
                        "error"
                    )
                    st.rerun()
                
                # Current game stats
                st.markdown(f"""
                <div class="current-stats">
                👤 Player: {st.session_state.player_name} | 
                💫 Score: {st.session_state.total_score} | 
                📝 Questions: {st.session_state.questions_asked + 1}/{st.session_state.game_length} |
                🎯 Topic: {st.session_state.topic}
                </div>
                """, unsafe_allow_html=True)
                
                # Question display
                st.markdown(f"""
                <div class="question-display">
                Question {st.session_state.questions_asked + 1}/{st.session_state.game_length}:
                {st.session_state.current_question["question"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Answer choices in two columns
                col1, col2 = st.columns(2)
                
                # First two answers (A and B)
                with col1:
                    for i in range(2):
                        if time_remaining > 0:
                            if st.button(st.session_state.current_question["choices"][i], 
                                       key=f"choice_{i}", 
                                       disabled=st.session_state.answer_selected,
                                       use_container_width=True):
                                time_remaining = check_answer(chr(65 + i))
                
                # Last two answers (C and D)
                with col2:
                    for i in range(2, 4):
                        if time_remaining > 0:
                            if st.button(st.session_state.current_question["choices"][i], 
                                       key=f"choice_{i}", 
                                       disabled=st.session_state.answer_selected,
                                       use_container_width=True):
                                time_remaining = check_answer(chr(65 + i))
                
                # Feedback area
                if st.session_state.feedback:
                    message, type_ = st.session_state.feedback
                    with st.container():
                        st.markdown(f"""
                        <div class="response-area">
                        {message}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("Next Question ➡️", type="primary", use_container_width=True):
                        st.session_state.questions_asked += 1
                        st.session_state.current_question = None
                        st.rerun()
            
        elif st.session_state.questions_asked >= st.session_state.game_length:
            # Display enhanced game over screen with all stats
            display_game_over(
                st.session_state.player_name,
                st.session_state.total_score,
                st.session_state.topic,
                st.session_state.questions_asked,
                st.session_state.game_length
            )

        # Footer section
        st.markdown("---")

        # Social sharing section with updated URLs
        app_url = "https://geminitriviaapp.streamlit.app/"
        share_text = "Challenge yourself with this Gemini-powered AI Trivia game!"

        # Define footer HTML
        footer_html = f"""
            <div class="footer-section">
                <div class="footer-content">
                    <span style="font-size: 1.2rem; font-weight: 600;">
                        Gemini GenAI Trivia Challenge
                    </span>
                    <div class="footer-social">
                        <a href="https://www.linkedin.com/shareArticle?mini=true&url={app_url}&title={share_text}"
                        class="social-button linkedin-button" target="_blank" rel="noopener noreferrer">
                            <i class="fab fa-linkedin"></i> LinkedIn
                        </a>
                        <a href="https://www.facebook.com/sharer/sharer.php?u={app_url}"
                        class="social-button facebook-button" target="_blank" rel="noopener noreferrer">
                            <i class="fab fa-facebook"></i> Facebook
                        </a>
                        <a href="https://twitter.com/intent/tweet?text={share_text}&url={app_url}"
                        class="social-button twitter-button" target="_blank" rel="noopener noreferrer">
                            <i class="fab fa-twitter"></i> X/Twitter
                        </a>
                        <a href="mailto:?subject=Check out Gemini GenAI Trivia Challenge!&body={share_text}%0A%0A{app_url}"
                        class="social-button email-button">
                            <i class="fas fa-envelope"></i> Email
                        </a>
                    </div>
                    <span style="font-size: 1rem;">
                        Designed by 
                        <a href="https://www.linkedin.com/in/lindsayhiebert/" target="_blank" 
                        style="text-decoration: none; color: #1E88E5;" rel="noopener noreferrer">
                        Lindsay Hiebert
                        </a>
                    </span>
                </div>
            </div>
        """

        # Display footer with error handling
        try:
            st.markdown(footer_html, unsafe_allow_html=True)

            # Footer image with error handling
            try:
                footer_img = load_and_resize_image("FooterImage.png", width=800)
                if footer_img:
                    aspect_ratio = 0.02
                    new_height = int(800 * aspect_ratio)
                    from PIL import Image
                    resized_img = footer_img.resize((1200, new_height), Image.Resampling.LANCZOS)
                    st.image(resized_img, use_container_width=True)
            except Exception as e:
                st.warning("Unable to load footer image")

            st.markdown("---")
        except Exception as e:
            st.error(f"Error displaying footer: {str(e)}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        print(f"Error in main(): {str(e)}")
        


def cleanup():
    """Cleanup function to handle graceful shutdown"""
    try:
        import grpc
        grpc.shutdown_all_channels()
    except:
        pass

# Register cleanup function
atexit.register(cleanup)



if __name__ == "__main__":
    initialize_session_state()
    try:
        main()
    except Exception as e:
        st.error("The application encountered a critical error. Please refresh the page or try again later.")
        print(f"Critical error: {str(e)}")
    finally:
        cleanup()