"use server";
import { ChatGroq } from "@langchain/groq";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import {
  Annotation,
  END,
  MemorySaver,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  ToolMessage,
} from "@langchain/core/messages";

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { toNamespacedPath } from "path/win32";

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
});

const tools = [
  new TavilySearchResults({
    maxResults: 5,
    apiKey: process.env.NEXT_PUBLIC_SEARCH_KEY,
  }),
];
const toolNode = new ToolNode(tools);
const picks = new ChatGroq({
  apiKey: process.env.NEXT_PUBLIC_GOOGLE_AI_KEY,
  model: "llama-3.3-70b-versatile",
  temperature: 0,
});

function shouldSearch(state: typeof GraphState.State): string {
  const { messages } = state;
  console.log("---DECIDE TO SEARCH---");
  const lastMessage = messages[messages.length - 1];

  if (
    "tool_calls" in lastMessage &&
    Array.isArray(lastMessage.tool_calls) &&
    lastMessage.tool_calls.length
  ) {
    console.log("---DECISION: SEARCH---");
    return "search";
  }

  return END;
}
function checkRelevance(state: typeof GraphState.State): string {
  console.log("---CHECK RELEVANCE---");

  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  if (!("tool_calls" in lastMessage)) {
    throw new Error(
      "The 'checkRelevance' node requires the most recent message to contain tool calls."
    );
  }
  const toolCalls = (lastMessage as AIMessage).tool_calls;
  if (!toolCalls || !toolCalls.length) {
    throw new Error("Last message was not a function message");
  }

  if (toolCalls[0].args.binaryScore === "yes") {
    console.log("---DECISION: DOCS RELEVANT---");
    return "yes";
  }
  console.log("---DECISION: DOCS NOT RELEVANT---");
  return "no";
}

async function gradeInputs(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GRADE USER INPUT STARTED---");
  const { messages } = state;

  // Define the tool schema
  const gtool = {
    name: "give_relevance_score",
    description:
      "Give a relevance score to the user input for TV show recommendations",
    schema: z.object({
      binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
    }),
  };

  // Create the prompt template
  const prompt = ChatPromptTemplate.fromTemplate(`
      You are an expert TV show recommendation assistant and prompt enigneer. Your task is to evaluate whether the user's input is relevant and sufficient for making a TV show recommendation.

      Analyze the following user input and determine if it provides enough information to make a quality recommendation. Consider:
      1. Does it describe specific genres, themes, or preferences?
      2. Does it mention any particular shows or characters they enjoy?
      3. Is it long enough to provide meaningful context?
      4. Does it avoid being too vague or generic?

      User Input: {input}

      Respond with either 'yes' if the input is sufficient for making a recommendation, or 'no' if it needs more detail or is too vague.
    `);

  // Bind the tool to the model with proper configuration
  const picksWithGrade = picks.bindTools([gtool], {
    tool_choice: {
      type: "function",
      function: { name: "give_relevance_score" },
    },
  });

  // Create and invoke the chain
  const chain = prompt.pipe(picksWithGrade);
  const score = await chain.invoke({
    input: messages[0].content as string,
  });

  return {
    messages: [score],
  };
}

async function rewrite(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---IMPROVE USER PROMPT STARTED---");
  const { messages } = state;

  const prompt = ChatPromptTemplate.fromTemplate(`
    You are an expert TV show recommendation assistant. Your task is to analyze user input and transform it into a clear, detailed question for precise recommendations. Think like the user to understand their needs and preferences.

    Carefully analyze the following user input, examining both explicit details and underlying semantic intent. Based on your analysis, reformulate the input into a more precise and actionable question for TV show recommendations.

    User Input: {input}

    Follow these guidelines when rewriting the question:
    1. Semantic Analysis: Identify explicit requests (genres, themes, show characteristics) and implicit preferences or desired outcomes
    2. Clarification: Add details to resolve vagueness or ambiguity, specifying preferences for era, style, pacing, or other relevant factors
    3. Context Enrichment: Incorporate contextual elements that could enhance recommendations (viewing time, experience goals, similar shows)
    4. Conciseness and Precision: Craft a question that is both concise and detailed, clearly outlining recommendation parameters
    5. Intent Preservation: Maintain the original input's core meaning while making it more targeted and actionable

    Improved Question:
  `);

  const chain = prompt.pipe(picks);
  const suggestions = await chain.invoke({
    input: messages[0].content as string,
  });
  return {
    messages: [suggestions],
  };
}

async function agent(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---CALL AGENT---");

  const { messages } = state;
  const filteredMessages = messages.filter((message) => {
    if (
      "tool_calls" in message &&
      Array.isArray(message.tool_calls) &&
      message.tool_calls.length > 0
    ) {
      console.log(message.tool_calls[0].name);
      return message.tool_calls[0].name !== "give_relevance_score";
    }
    return true;
  });

  const recAgent = picks.bindTools(tools);
  const response = await recAgent.invoke(filteredMessages);
  return {
    messages: [response],
  };
}

async function cleanupBeforeSearch(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  const { messages } = state;
  // Filter out messages with give_relevance_score tool calls
  const cleanedMessages = messages.filter((message) => {
    if ("tool_calls" in message && Array.isArray(message.tool_calls)) {
      return !message.tool_calls.some(
        (call) => call.name === "give_relevance_score"
      );
    }
    return true;
  });

  return {
    messages: cleanedMessages,
  };
}

async function generate(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE---");

  const { messages } = state;
  const question = messages[0].content as string;
  // Extract the most recent ToolMessage
  const lastToolMessage = messages
    .slice()
    .reverse()
    .find((msg) => msg.getType() === "tool");
  if (!lastToolMessage) {
    throw new Error("No tool message found in the conversation history");
  }

  const rec = lastToolMessage.content as string;

  const prompt = ChatPromptTemplate.fromTemplate(`
      You are an expert TV show recommendation assistant. Your task is to generate personalized recommendations based on the user's preferences and the searched context.

      User's question: {question}
      
      
      context:{context}

      Instructions:
      1. Analyze the user's preferences carefully
      2. Consider the context from our knowledge base
      3. Generate 3-5 specific TV show recommendations in the best possible order of watch in a list format
      4. For each recommendation, include:
         - Title
       
        - streaming platform (e.g., Netflix, Hulu, Amazon Prime)
        dont add extract explanation just the list
        format: title on streaming platform
      5. Use a friendly and engaging tone
      6. If no good matches are found, suggest alternative approaches
      Answer:
    `);
  const chain = prompt.pipe(picks);
  const response = await chain.invoke({ context: rec, question });
  return {
    messages: [response],
  };
}

const workflow = new StateGraph(GraphState)
  .addNode("agent", agent)
  .addNode("search", toolNode)
  .addNode("gradeInputs", gradeInputs)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate)
  .addNode("cleanup", cleanupBeforeSearch);

// Modify the edges to include cleanup
workflow.addEdge(START, "agent");
workflow
  .addEdge("agent", "gradeInputs")
  .addConditionalEdges("gradeInputs", checkRelevance, {
    yes: "cleanup", // Go to cleanup first
    no: "rewrite",
  });
workflow.addEdge("cleanup", "search"); // Then to search
workflow.addEdge("search", "generate");
workflow.addEdge("generate", END);
workflow.addEdge("rewrite", "agent");

const app = workflow.compile({});

export async function watch(userInput: string) {
  const config = { configurable: { thread_id: "conversation-2" } };
  const inputs = {
    messages: [new HumanMessage(userInput)],
  };
  let finalState;
  for await (const output of await app.stream(inputs)) {
    for (const [key, value] of Object.entries(output)) {
      const lastMsg = output[key].messages[output[key].messages.length - 1];
      console.log(`Output from node: '${key}'`);
      console.dir(
        {
          type: lastMsg._getType(),
          content: lastMsg.content,
          tool_calls: lastMsg.tool_calls,
        },
        { depth: null }
      );
      console.log("---\n");
      finalState = value;
    }
  }
  console.log(JSON.stringify(finalState, null, 2));
  const message = JSON.stringify(finalState, null, 2);
  return message;
}
