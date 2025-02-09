"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Loader2 } from "lucide-react";

// Import the picks function from your LangGraph workflow module.
import { watch } from "@/lib/agent"; // adjust the path as needed
import { json } from "stream/consumers";

// Define the form schema.
const formSchema = z.object({
  showPreferences: z
    .string()
    .min(10, "Description must be at least 10 characters")
    .max(500, "Description must be less than 500 characters"),
});

export default function TVShowRecommenderRAGHITL() {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      showPreferences: "",
    },
  });

  // Define state for recommendation, loading, and error messages.
  const [recommendation, setRecommendation] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // onSubmit handler: calls the picks function with the TV show preferences.
  // onSubmit handler: calls the picks function with the TV show preferences.
  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    setLoading(true);
    setError(null);
    setRecommendation(null);
    const extractContent = (jsonResponse: unknown) => {
      const parsedResponse =
        typeof jsonResponse === "string"
          ? JSON.parse(jsonResponse)
          : jsonResponse;

      return parsedResponse.messages[0].kwargs.content;
    };
    try {
      console.log(values.showPreferences);
      const res = await watch(values.showPreferences);
      console.log(res);
      const rec = JSON.parse(res);
      console.log(rec);
      if (res) {
        const recommendation = extractContent(res);
        setRecommendation(recommendation);
      } else {
        setRecommendation("No recommendation received.");
      }

      form.reset();
    } catch (err: any) {
      console.error("Error in picks:", err);
      setError(
        err.message || "An error occurred while fetching the recommendation."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen w-full flex-col">
      <main className="flex flex-1 flex-col items-center justify-center p-4">
        <Card className="w-full max-w-2xl border-none">
          <CardHeader>
            <CardTitle>Watch by picks</CardTitle>
          </CardHeader>
          <CardContent>
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="space-y-4"
              >
                <FormField
                  control={form.control}
                  name="showPreferences"
                  render={({ field }) => (
                    <FormItem>
                      <FormControl>
                        <Textarea
                          placeholder="Describe your TV show preferences (e.g., 'I like sci-fi shows with complex plots and strong character development')"
                          className="min-h-[100px]"
                          disabled={loading}
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <Button type="submit" className="w-full" disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />{" "}
                      Loading...
                    </>
                  ) : (
                    "Get Recommendation"
                  )}
                </Button>
              </form>
            </Form>

            {/* Render error message if any */}
            {error && (
              <Alert variant="destructive" className="mt-4">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Render recommendation if available */}
            {recommendation && (
              <Alert variant="default" className="mt-4">
                <AlertTitle>Recommendation</AlertTitle>
                <AlertDescription>{recommendation}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
