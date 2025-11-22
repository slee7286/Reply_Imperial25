
# GeminiFit — AI-Powered Personal Trainer & Workout Planner

**GeminiFit** is a sleek React application that leverages Google GenAI to deliver AI-powered fitness coaching. Users can chat with an AI coach to generate personalized, structured workout plans, participate in live voice sessions, and track their progress via a clean dashboard.

---

## Features

- **Dashboard View**  
  Overview of your fitness journey and app status in a modern, responsive layout.

- **AI Coach Chat**  
  Conversational interface with an AI personal trainer that understands your fitness goals and generates detailed, table-formatted workout plans using Google GenAI.

- **Live Session**  
  Real-time voice interaction mode for hands-free coaching and guidance.

- **Responsive UI**  
  Adaptive sidebar and mobile-friendly bottom navigation for smooth navigation on all device sizes.

- **Workout Plan Viewer**  
  View rich, multi-day workout plans with detailed exercises, sets, reps, rest, and notes in a modal overlay.

- **Grounding Sources**  
  Display relevant grounding links and references provided by the AI model for transparency and learning.

---

## Tech Stack & Libraries

- **React** with hooks and functional components  
- **TypeScript** for strict typing and type safety  
- **Lucide React** for crisp SVG icons  
- **Tailwind CSS** for rapid, utility-first styling  
- **Google GenAI** (`@google/genai`) for AI chat and function-calling  
- **Modern React features:** useState, useEffect, useRef, conditional rendering  

---

## Project Structure

- **`App.tsx`**  
  Root component managing the app state and main navigation between views: Dashboard, Chat, and Live Session.

- **`components/ChatInterface.tsx`**  
  The AI chat interface where users interact with the chatbot, send messages, and receive AI-generated workout plans. Supports function calling to Google GenAI’s structured workout plan generation tool.

- **`components/LiveSession.tsx`**  
  (Not included here) Handles voice-based live coaching sessions.

- **`components/Dashboard.tsx`**  
  (Not included here) Displays user overview and app stats.

- **`types.ts`**  
  Contains TypeScript enums and interfaces like `AppView`, `Message`, and `WorkoutPlan`.

- **`constants.ts`**  
  Stores AI model names, system instructions, and other static configs.

---

## How It Works

1. **Navigation**  
   Users switch between Dashboard, Chat, and Live Session via sidebar (desktop) or bottom nav (mobile).

2. **Chat Interface**  
   - User types a fitness goal or question and sends a message.  
   - The app sends chat history and the user message to Google GenAI chat API configured with a custom workout plan tool.  
   - Google GenAI may respond with a function call to generate a workout plan, returning structured JSON describing plan name, goals, schedule, exercises, sets, reps, rest, and notes.  
   - The app parses this response and displays the plan summary with a "View Plan" button.  
   - Clicking "View Plan" opens a modal with a detailed workout plan table broken down by days and exercises.

3. **Live Session**  
   - Users can interact via voice input for a more immersive coaching experience (implementation not shown here).

4. **Grounding Links**  
   - When available, the AI’s response includes relevant source links, displayed below the chat for transparency.

---

## Installation

```bash
# Clone repo
git clone https://github.com/yourusername/geminifit.git
cd geminifit

# Install dependencies
npm install

# Start development server
npm start
```

---

## Environment Variables

Create a `.env` file in the root and add:

```env
REACT_APP_API_KEY=your_google_genai_api_key_here
```

This API key is used internally by the `GoogleGenAI` client to authenticate requests.

---

## Usage

- Launch the app locally or deploy it to your favorite static hosting service.
- Navigate to the **Chat Coach** view and type requests like:  
  - "Create a 3-day hypertrophy workout plan"  
  - "I want a beginner weight loss routine"  
- View generated workout plans in the modal and follow the exercises.
- Use the **Live Session** for voice-based interaction.
- Monitor overall progress in the **Dashboard**.

---

## Customization & Extensibility

- **Add More AI Tools:** Extend the Google GenAI tool list with custom function declarations.  
- **Improve UI:** Customize Tailwind config or replace UI components.  
- **Add Authentication:** Connect with auth providers for user accounts.  
- **Persist Data:** Integrate with backend or local storage for saving workout history.

---

## Dependencies

- React  
- @google/genai  
- lucide-react  
- Tailwind CSS  
- Typescript

---

## License

MIT License

---

## Contact

Created by [Your Name] — feel free to open issues or pull requests.

---

**GeminiFit** harnesses the power of AI to make personal fitness coaching accessible and interactive. Train smart, stay motivated!

