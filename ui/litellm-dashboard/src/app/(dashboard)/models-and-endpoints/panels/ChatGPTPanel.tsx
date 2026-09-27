"use client";

import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, ExternalLink } from "lucide-react";
import { apiClient } from "@/components/networking";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type ConnectionStatus = { signed_in: boolean };
type DeviceCodeResponse = {
  user_code: string;
  verification_uri: string;
  interval: number;
};

export default function ChatGPTPanel({ accessToken }: { accessToken: string | null }) {
  const [status, setStatus] = useState<ConnectionStatus | null>(null);
  const [deviceFlow, setDeviceFlow] = useState<DeviceCodeResponse | null>(null);
  const [error, setError] = useState("");
  const [account, setAccount] = useState("default");
  const [activeAccount, setActiveAccount] = useState("default");
  const [loading, setLoading] = useState(true);
  const [requesting, setRequesting] = useState(false);

  const loadStatus = useCallback(async () => {
    try {
      const connection = await apiClient.get<ConnectionStatus>("/chatgpt/status", { accessToken, query: { account: activeAccount } });
      setStatus(connection);
      if (connection.signed_in) {
        setDeviceFlow(null);
      }
      return connection.signed_in;
    } catch (reason) {
      // ignore
      return false;
    }
  }, [accessToken, activeAccount]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      loadStatus()
        .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)))
        .finally(() => setLoading(false));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [loadStatus]);

  // Poll for token completion if in device flow
  useEffect(() => {
    if (!deviceFlow || status?.signed_in) return;

    const interval = setInterval(async () => {
      const signedIn = await loadStatus();
      if (signedIn) {
        clearInterval(interval);
      }
    }, 3000); // 3 seconds

    return () => clearInterval(interval);
  }, [deviceFlow, status?.signed_in, loadStatus]);

  const startLogin = async () => {
    setRequesting(true);
    setError("");
    try {
      const resp = await apiClient.post<DeviceCodeResponse>("/chatgpt/login", { accessToken, query: { account } });
      setDeviceFlow(resp);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRequesting(false);
    }
  };

  if (loading) return <p className="text-sm text-muted-foreground">Checking ChatGPT connection…</p>;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-3">
            <div>
              <CardTitle>ChatGPT (Codex) OAuth</CardTitle>
              <CardDescription>Connect OpenAI directly to this LiteLLM proxy using device code flow.</CardDescription>
            </div>
            {status?.signed_in && (
              <Badge variant="secondary" className="gap-1">
                <CheckCircle2 className="size-3" /> Connected
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <p className="text-sm text-destructive" role="alert">
              {error}
            </p>
          )}

          {!status?.signed_in && !deviceFlow && (
            <>
                <div>
                  <label className="mb-1 block text-sm font-medium">Account Profile Name</label>
                  <input className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background mb-4" value={account} onChange={(e) => { setAccount(e.target.value); setActiveAccount(e.target.value); }} placeholder="default" />
                </div>
                <Button onClick={startLogin} disabled={requesting} className="gap-2 bg-[#10a37f] hover:bg-[#0e906f]">
                  Sign in with ChatGPT
                </Button>
            </>
          )}
          {!status?.signed_in && deviceFlow && (
            <div className="rounded-md border p-4 bg-muted/30">
                <h3 className="font-semibold mb-2 text-lg">Verification Required</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  1. Open the verification link below in your browser.<br />
                  2. Enter the code shown below to authorize.
                </p>
                <div className="text-center my-4 font-mono text-3xl font-bold tracking-widest bg-white p-3 rounded border">
                  {deviceFlow.user_code}
                </div>
                <div className="text-center mt-4">
                  <a href={deviceFlow.verification_uri} target="_blank" rel="noreferrer">
                    <Button variant="outline" className="gap-2 w-full">
                      Open Verification Page <ExternalLink className="size-4" />
                    </Button>
                  </a>
                </div>
                <p className="text-sm text-muted-foreground mt-4 text-center animate-pulse">
                  Waiting for authorization...
                </p>
            </div>
          )}
          {status?.signed_in && (
            <div className="space-y-2">
              <div className="mb-4 bg-muted/50 p-2 rounded-md">
                <span className="text-sm font-semibold">Active Profile:</span> <span className="font-mono text-sm">{activeAccount}</span>
                <Button variant="link" size="sm" onClick={() => { setActiveAccount("default"); setStatus(null); }}>Switch</Button>
              </div>
              <p className="text-sm text-muted-foreground">
                Your ChatGPT device is authorized. You can now use <code>chatgpt/*</code> models in your routing.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
