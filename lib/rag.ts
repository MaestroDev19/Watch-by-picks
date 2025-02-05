"use server";

import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createBrowserClient } from "@/supabase/client";
import { Annotation, START, StateGraph } from "@langchain/langgraph";
import { createRetrieverTool } from "langchain/tools/retriever";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { END } from "@langchain/langgraph";
import { pull } from "langchain/hub";
import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";

const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.NEXT_PUBLIC_GOOGLE_AI_KEY,
  model: "gemini-2.0-flash-exp",
  temperature: 0,
  maxOutputTokens: 2025,
});

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
});

const tool = [
  new TavilySearchResults({
    maxResults: 5,
    apiKey: process.env.NEXT_PUBLIC_SEARCH_KEY,
  }),
];

const searchToolNode = new ToolNode<typeof GraphState.State>(tool);

function shouldRetrieve(state: typeof GraphState.State): string {
  const { messages } = state;
  console.log("---DECIDE TO SEARCH---");
  const lastMessage = messages[messages.length - 1];
  if (
    "tool_calls" in lastMessage &&
    Array.isArray(lastMessage.tool_calls) &&
    lastMessage.tool_calls.length > 0
  ) {
    console.log("---DECISION: SEARCH---");
    return "search";
  }
  return END;
}

async function search(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---STARTING SEARCH---");
  const { messages } = state;

  const tool = new TavilySearchResults({
    maxResults: 5,
    apiKey: process.env.NEXT_PUBLIC_SEARCH_KEY,
  });

  const prompt = ChatPromptTemplate.fromTemplate(`
    You are an expert TV show recommendation assistant with deep knowledge of television content across all genres and eras. Your primary task is to:

    1. Analyze the user's request and understand their intent
    2. Evaluate the relevance of retrieved documents based on:
       - Content match (title, description, cast, etc.)
       - Genre alignment
       - Release year/time period
       - Popularity and ratings
       - Cultural significance
   
    User's Request:
    {question}

 
`);
  const search = model.bindTools([tool], { tool_choice: tool.name });
  const chain = prompt.pipe(search);
  const searchResult = await chain.invoke({
    question: messages[0].content as string,
  });
  return {
    messages: [searchResult],
  };
}
async function gradeRecommendation(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GET RELEVANCE---");
  const { messages } = state;
  const tool = {
    name: "give_relevance_score",
    description:
      "Evaluate the retrieved documents to determine how well they support the user's recommendation request. Return a detailed relevance score between 0 (no meaningful connection) and 100 (perfect match) with justification. Use these scoring guidelines: 90-100 = Perfect match, directly answers query; 70-89 = Strong match, covers most aspects; 50-69 = Partial match, some relevant information; 30-49 = Weak match, limited relevance; 0-29 = No meaningful connection.",
    schema: z.object({
      relevanceScore: z
        .number()
        .min(0)
        .max(100)
        .describe(
          "A detailed relevance score between 0 and 100 indicating how well the document matches the query. " +
            "Scoring Guidelines: " +
            "90-100 = Perfect match, directly answers query; " +
            "70-89 = Strong match, covers most aspects; " +
            "50-69 = Partial match, some relevant information; " +
            "30-49 = Weak match, limited relevance; " +
            "0-29 = No meaningful connection"
        ),
      explanation: z
        .string()
        .optional()
        .describe("A single sentence justification for the relevance score,"),
    }),
  };
  const prompt = ChatPromptTemplate.fromTemplate(`
        You are an expert TV show recommendation assistant. Your task is to evaluate how relevant the retrieved documents are to the user's request and provide a detailed relevance score with justification.

        Here's the user's request:
        {question}

        Here are the retrieved documents:
        {context}

        Please analyze the documents and provide:
        1. A detailed relevance score between 0 and 100 using these guidelines:
           - 90-100: Perfect match, directly answers the query
           - 70-89: Strong match, covers most aspects
           - 50-69: Partial match, some relevant information
           - 30-49: Weak match, limited relevance
           - 0-29: No meaningful connection
        2. A clear justification explaining why you gave this score

        Consider these factors in your evaluation:
        - Does the document contain information directly related to the user's query?
        - Is the information up-to-date and accurate?
        - Does the document provide sufficient detail to be helpful?
        - Are there multiple documents that support the same recommendation?
        - How well does the document align with the user's preferences and requirements?

        Return your response in JSON format with these fields:
        - relevanceScore: number (0-100)
        - explanation: string (single sentence justification for the score`);

  const relavance = model.bindTools([tool], { tool_choice: tool.name });
  const chain = prompt.pipe(relavance);
  const lastMessage = messages[messages.length - 1];

  const score = await chain.invoke({
    question: messages[0].content as string,
    context: lastMessage.content as string,
  });

  return {
    messages: [score],
  };
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

  // Extract the relevanceScore from the tool call arguments
  const relevanceScore = toolCalls[0].args.relevanceScore;
  console.log(`Relevance Score: ${relevanceScore}`);

  if (typeof relevanceScore !== "number") {
    throw new Error("Expected relevanceScore to be a number.");
  }

  // Use the scoring guidelines to determine relevance
  if (relevanceScore >= 70) {
    console.log("---DECISION: DOCS RELEVANT (Strong/Perfect Match)---");
    return "yes";
  } else if (relevanceScore >= 50) {
    console.log("---DECISION: DOCS PARTIALLY RELEVANT (Partial Match)---");
    return "no"; // Consider refining search even for partial matches
  } else {
    console.log("---DECISION: DOCS NOT RELEVANT (Weak/No Match)---");
    return "no";
  }
}

async function agent(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---CALL AGENT (Recommendation Mode)---");

  const { messages } = state;

  const filteredMessages = messages.filter((message) => {
    if (
      "tool_calls" in message &&
      Array.isArray(message.tool_calls) &&
      message.tool_calls.length > 0
    ) {
      return message.tool_calls[0].name !== "give_relevance_score";
    }
    return true;
  });
  const tool = new TavilySearchResults({
    maxResults: 5,
    apiKey: process.env.NEXT_PUBLIC_SEARCH_KEY,
  });

  // Bind the tools (both internal retriever and external search) to the model.
  const agentModel = new ChatGoogleGenerativeAI({
    apiKey: process.env.NEXT_PUBLIC_GOOGLE_AI_KEY,
    model: "gemini-2.0-flash-exp",
    temperature: 0,
    maxOutputTokens: 2025,
    streaming: true,
  }).bindTools([tool]);

  try {
    // Invoke the agent model with the filtered conversation history.
    const response = await agentModel.invoke(filteredMessages);
    console.log("---AGENT RESPONSE GENERATED---");
    return { messages: [response] };
  } catch (error) {
    console.error("Error invoking the agent:", error);
    throw new Error(
      "Agent invocation failed. Please check your tools and configuration."
    );
  }
}

async function rewrite(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---REWRITE FUNCTION: Refining the recommendation query---");

  const { messages } = state;
  const question = messages[0].content as string;

  // Create a prompt that instructs the LLM to improve the user's recommendation query
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a highly experienced TV show recommendation assistant.
The user has asked:
"{question}"

Please analyze if the user is asking for:
1. A specific show recommendation
2. Recommendations based on 
- Content match (title, description, cast, etc.)
       - Genre alignment
       - Release year/time period
       - Popularity and ratings
       - Cultural significance
3. Help understanding which shows might suit their preferences

Based on the analysis, refine the query to be more precise and actionable for fetching relevant TV show recommendations with good rating.
Consider including specific details such as desired genre, mood, style, or any other factors that might improve the search accuracy.
If the query is already clear and specific, return it as is.
Return only the improved query as plain text. `
  );

  const queryModel = new ChatGoogleGenerativeAI({
    apiKey: process.env.NEXT_PUBLIC_GOOGLE_AI_KEY,
    model: "gemini-2.0-flash-exp",
    temperature: 0,
    maxOutputTokens: 2025,
    streaming: true,
  });
  try {
    const response = await prompt.pipe(queryModel).invoke({ question });
    console.log("---REWRITE FUNCTION: Query refined successfully---");
    return { messages: [response] };
  } catch (error) {
    console.error("Error in rewrite function:", error);
    throw new Error("Rewrite function failed to generate a refined query.");
  }
}
async function generate(
  state: typeof GraphState.State
): Promise<Partial<typeof GraphState.State>> {
  console.log("---GENERATE: Generating final recommendation---");

  const { messages } = state;
  const question = messages[0].content as string;

  // Extract the most recent tool message which contains the retrieved documents or search results.
  const lastToolMessage = messages
    .slice()
    .reverse()
    .find((msg) => msg._getType() === "tool");
  if (!lastToolMessage) {
    throw new Error("No tool message found in the conversation history");
  }

  // 'retrievedDocs' should contain the context (e.g. TV show details or search results) used for recommendations.
  const retrievedDocs = lastToolMessage.content as string;
  console.log("Retrieved context for recommendation:", retrievedDocs);

  // Pull the RAG prompt template that is tailored for recommendation generation.
  const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");

  // Initialize the generative model using Google Generative AI with streaming enabled.
  const llm = new ChatGoogleGenerativeAI({
    apiKey: process.env.NEXT_PUBLIC_GOOGLE_AI_KEY,
    model: "gemini-2.0-flash-exp",
    temperature: 0,
    maxOutputTokens: 2025,
    streaming: true,
  });

  // Pipe the prompt into the generative model to form the RAG chain.
  const ragChain = prompt.pipe(llm);

  // Invoke the chain using the retrieved context and original user question.
  // The prompt template should be designed to instruct the model to produce a clear recommendation.
  const response = await ragChain.invoke({
    context: retrievedDocs,
    question,
  });

  console.log("---GENERATE: Recommendation generated successfully---");
  return {
    messages: [response],
  };
}

const workflow = new StateGraph(GraphState)
  .addNode("agent", agent)
  .addNode("search", searchToolNode)
  .addNode("gradeRecommendation", gradeRecommendation)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate);

workflow.addEdge(START, "agent");

workflow.addConditionalEdges("agent", shouldRetrieve);

workflow.addEdge("search", "gradeRecommendation");

workflow.addConditionalEdges("gradeRecommendation", checkRelevance, {
  yes: "generate", // Documents are relevant
  no: "rewrite", // Documents need refinement
});

workflow.addEdge("rewrite", "agent");
workflow.addEdge("generate", END);

const app = workflow.compile();

export async function picks(input: string) {
  const inputs = {
    messages: [new HumanMessage(input)],
  };

  let finalState;
  try {
    for await (const output of await app.stream(inputs)) {
      for (const [key, value] of Object.entries(output)) {
        const messages = output[key]?.messages;

        if (!messages || messages.length === 0) {
          console.warn(`No messages found in output from node: '${key}'`);
          continue;
        }

        const lastMsg = messages[messages.length - 1];

        if (!lastMsg) {
          console.warn(
            `Last message is undefined in output from node: '${key}'`
          );
          continue;
        }

        console.log(`Output from node: '${key}'`);
        console.dir(
          {
            type: lastMsg._getType?.(),
            content: lastMsg.content,
            tool_calls: lastMsg.tool_calls,
          },
          { depth: null }
        );
        console.log("---\n");
        finalState = value;
      }
    }

    if (finalState) {
      return JSON.stringify(finalState, null, 2);
    } else {
      console.warn("No final state generated");
    }
  } catch (error) {
    console.error("Error in picks function:", error);
    const errorMessage =
      error instanceof Error ? error.message : "An unknown error occurred";
    throw new Error(`Failed to generate recommendations: ${errorMessage}`);
  }
}
