# Watch - Personalized TV Show Recommendation System

## Overview

Watch is an intelligent TV show recommendation system powered by LangChain and Groq's language models. It provides personalized TV show suggestions based on user preferences, leveraging advanced natural language processing and search capabilities.

## Features

- **Personalized Recommendations**: Get tailored TV show suggestions based on your preferences
- **Intelligent Prompt Analysis**: The system analyzes and enhances user input for better recommendations
- **Search Integration**: Utilizes Tavily search for up-to-date show information
- **Responsive UI**: Built with modern React components and Tailwind CSS
- **Scalable Architecture**: Designed with modular components for easy extension

## Tech Stack

- **Frontend**: Next.js 15, React 19, Tailwind CSS
- **AI**: LangChain, Groq API
- **Search**: Tavily Search API
- **UI Components**: Radix UI, Shadcn UI
- **State Management**: React Hook Form
- **Type Safety**: TypeScript, Zod

## Getting Started

### Prerequisites

- Node.js v18+
- Yarn or npm
- Groq API key
- Tavily Search API key

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/watch.git
cd watch
```

2. Install dependencies:

```bash
yarn install
# or
npm install
```

3. Create a `.env` file in the root directory with the following variables:

```env
NEXT_PUBLIC_GOOGLE_AI_KEY=your_groq_api_key
NEXT_PUBLIC_SEARCH_KEY=your_tavily_api_key
```

4. Run the development server:

```bash
yarn dev
# or
npm run dev
```

## Project Structure

```
watch/
├── app/                  # Next.js app directory
│   ├── layout.tsx        # Root layout
│   └── page.tsx          # Main page component
├── components/           # UI components
├── hooks/                # Custom hooks
├── lib/                  # Utility functions and AI logic
│   └── agent.ts          # LangChain agent implementation
├── public/               # Static assets
└── styles/               # Global styles
```

## Key Components

### AI Agent (lib/agent.ts)

The core recommendation logic is implemented in the `agent.ts` file. It includes:

- State management with LangGraph
- Input grading and enhancement
- Search integration
- Recommendation generation

```typescript:lib/agent.ts
startLine: 1
endLine: 305
```

### Main Page (app/page.tsx)

The main page contains the user interface for inputting preferences and displaying recommendations.

```typescript:app/page.tsx
startLine: 1
endLine: 146
```

## API Documentation

### `watch(userInput: string)`

Main function that processes user input and returns recommendations.

**Parameters:**

- `userInput`: String containing the user's TV show preferences

**Returns:**

- JSON object containing the recommendation

**Example Usage:**

```typescript
const recommendation = await watch("I like sci-fi shows with complex plots");
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain for the AI framework
- Groq for the language model
- Tavily for search integration
- Shadcn UI for the component library

```

This README provides a comprehensive overview of the project, including setup instructions, key components, and contribution guidelines. It's designed to help both users and developers understand and work with the codebase effectively.
```
