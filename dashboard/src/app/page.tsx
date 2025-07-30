"use client"
import { useState } from "react";
import Head from "next/head";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  async function handleRun() {
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("http://localhost:8000/run-strategy", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Failed to run strategy", err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Head>
        <title>Quant Strategy Dashboard</title>
      </Head>

      <main className="min-h-screen p-6 bg-gray-50">
        <h1 className="text-2xl font-bold mb-4">Quant Strategy Generator</h1>

        <Textarea
          placeholder="Enter your strategy prompt here..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full max-w-2xl mb-4"
        />

        <Button onClick={handleRun} disabled={loading}>
          {loading ? "Running..." : "Run Strategy"}
        </Button>

        <div className="mt-8">
          {loading && (
            <div className="space-y-4">
              <Skeleton className="h-6 w-1/2" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-5/6" />
            </div>
          )}

          {result && (
            <Card className="mt-4 max-w-3xl">
              <CardContent className="p-4 space-y-2">
                <h2 className="text-lg font-semibold">{result.strategy.strategy_name}</h2>
                <p className="text-sm text-muted-foreground">{result.strategy.description}</p>
                <div className="pt-2">
                  <strong>Sharpe:</strong> {result.results.sharpe_ratio} | <strong>Win Rate:</strong> {result.results.win_rate} | <strong>Avg Return:</strong> {result.results.average_return}
                </div>
                <div className="pt-2">
                  <strong>Explanation:</strong>
                  <p className="text-sm mt-1 whitespace-pre-line">
                    {result.explanation}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </>
  );
}
