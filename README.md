Here's an updated `README.md` file for your Gemini GenAI Trivia App, reflecting the transition from OpenAI to Google Gemini models and incorporating the new app details:

---

# 🧠 Google Gemini GENAI Trivia Challenge

An advanced trivia game powered by Google's Gemini AI models that generates unique, challenging questions while preventing duplicates and ensuring high-quality content. Test your knowledge across any topic while racing against time!

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://geminitriviaapp.streamlit.app)

---

## ✨ Features

- **Smart Question Generation**:
  - Leverages Google Gemini for creating unique, engaging, and factually accurate questions.
  - Advanced validation ensures all answer choices are distinct and meaningful.
  - Question caching prevents duplicates during gameplay sessions.
  - Improved fact-checking and answer verification.

- **Flexible Game Modes**:
  - Choose between 5 or 10 question games.
  - Select any topic for customized trivia experiences.
  - Questions adapt to your chosen topic with domain-specific expertise.

- **Advanced Scoring System**:
  - Real-time scoring based on answer speed.
  - Timer with visual countdown.
  - Progress tracking throughout the game.
  - Detailed end-game statistics.

- **Enhanced User Interface**:
  - Clean, responsive design with intuitive controls.
  - Visual timer with progress bar.
  - Instant feedback with detailed explanations.
  - Mobile-friendly layout.

- **Comprehensive Leaderboard System**:
  - Global rankings across all topics.
  - Topic-specific leaderboards.
  - Historical score tracking.
  - Detailed player statistics.

- **Session Management**:
  - Question caching across games.
  - Persistent player statistics.
  - Option to reset cache and start fresh.
  - Seamless game transitions.

---

## 🚀 Quick Start

1. Visit [Gemini Trivia App](https://geminitriviaapp.streamlit.app/).
2. Enter your name and choose a topic.
3. Select game length (5 or 10 questions).
4. Start playing and race against time!
5. Learn from detailed explanations after each answer.

---

## 💻 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lhiebert01/geminitriviaapp.git
   cd geminitriviaapp
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

1. Create a `.env` file:
   ```bash
   cp env.template .env
   ```

2. Configure your API keys:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   ```

3. Set up Google Sheets integration:
   - Add your OAuth credentials JSON file.
   - Configure sheet access permissions.
   - Update the sheet URL in settings.

---

## 🛠️ Technical Features

- **Question Generation Engine**:
  - Advanced prompt engineering for consistent quality.
  - Multiple validation layers for answer uniqueness.
  - Sophisticated caching mechanism.
  - Optimized API usage.

- **Performance Optimizations**:
  - Efficient caching strategies.
  - Streamlined data management.
  - Response validation and error handling.
  - Rate limiting and API optimization.

- **Data Management**:
  - Secure Google Sheets integration.
  - Efficient leaderboard caching.
  - Real-time score updates.
  - Data persistence across sessions.

---

## 🚀 Deployment

The app is deployed on Streamlit Cloud and automatically syncs with the GitHub repository:

1. Main App: [Gemini Trivia App](https://geminitriviaapp.streamlit.app/)
2. Source Code: [GitHub Repository](https://github.com/lhiebert01/geminitriviaapp)

---

## 👥 Contributing

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.
4. Follow coding standards.
5. Add tests for new features.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

Developed by [Lindsay Hiebert](https://www.linkedin.com/in/lindsayhiebert/).

---