"use client";

import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, ExternalLink, RefreshCw } from "lucide-react";
import { apiClient, proxyBaseUrl } from "@/components/networking";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

type ConnectionStatus = { signed_in: boolean; project_id: string };
type ModelCatalog = { models: string[] };
type Quota = {
  id: string;
  remaining_percentage: number;
  reset_at: string | null;
  unlimited: boolean;
  display_name: string | null;
};
type Usage = { plan: string; quotas: Quota[]; checked_at: string };

const formatQuota = (quota: Quota): string => {
  if (quota.unlimited) return "Unlimited";
  return `${quota.remaining_percentage.toFixed(1)}% remaining`;
};

export default function AntigravityPanel({ accessToken }: { accessToken: string | null }) {
  const [status, setStatus] = useState<ConnectionStatus | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [projectId, setProjectId] = useState("");
  const [error, setError] = useState("");
  const [account, setAccount] = useState("default");
  const [activeAccount, setActiveAccount] = useState("default");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async (forceUsage = false) => {
    setError("");
    const connection = await apiClient.get<ConnectionStatus>("/antigravity/status", { accessToken, query: { account: activeAccount } });
    setStatus(connection);
    setProjectId(connection.project_id);
    if (!connection.signed_in || !connection.project_id) return;
    const [catalog, quota] = await Promise.all([
      apiClient.get<ModelCatalog>("/antigravity/models", { accessToken, query: { account: activeAccount } }),
      apiClient.get<Usage>("/antigravity/usage", { accessToken, query: { refresh: forceUsage, account: activeAccount } }),
    ]);
    setModels(catalog.models);
    setUsage(quota);
  }, [accessToken, activeAccount]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      load()
        .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)))
        .finally(() => setLoading(false));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const saveProject = async () => {
    setRefreshing(true);
    setError("");
    try {
      await apiClient.post("/antigravity/project", { accessToken, query: { account: activeAccount }, body: { project_id: projectId.trim() } });
      await load(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRefreshing(false);
    }
  };

  const refresh = async () => {
    setRefreshing(true);
    try {
      await load(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRefreshing(false);
    }
  };

  const returnTo = "/ui/models-and-endpoints?antigravity=connected";
  const loginAction = `${proxyBaseUrl ?? ""}/antigravity/login?account=${encodeURIComponent(account)}&return_to=${encodeURIComponent(returnTo)}`;

  if (loading) return <p className="text-sm text-muted-foreground">Checking Antigravity connection…</p>;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-3">
            <div>
              <CardTitle>Antigravity OAuth</CardTitle>
              <CardDescription>Connect Google Antigravity directly to this LiteLLM proxy.</CardDescription>
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
          {!status?.signed_in ? (
            <>
              <div>
                <label className="mb-1 block text-sm font-medium">Account Profile Name</label>
                <Input value={account} onChange={(e) => { setAccount(e.target.value); setActiveAccount(e.target.value); }} placeholder="default" className="mb-3" />
              </div>
              <form action={loginAction} method="post">
                <Button type="submit" className="gap-2">
                  Sign in with Google <ExternalLink className="size-4" />
                </Button>
              </form>
            </>
          ) : (
            <>
              <div className="mb-4 bg-muted/50 p-2 rounded-md">
                <span className="text-sm font-semibold">Active Profile:</span> <span className="font-mono text-sm">{activeAccount}</span>
                <Button variant="link" size="sm" onClick={() => { setActiveAccount("default"); setStatus(null); }}>Switch</Button>
              </div>
              <div>
                <label htmlFor="antigravity-project" className="mb-1 block text-sm font-medium">
                  Google Cloud project
                </label>
                <Input
                  id="antigravity-project"
                  value={projectId}
                  onChange={(event) => setProjectId(event.target.value)}
                  placeholder="Project ID"
                />
              </div>
              <div className="flex gap-2">
                <Button onClick={saveProject} disabled={refreshing}>
                  Discover or save project
                </Button>
                <Button variant="outline" onClick={refresh} disabled={refreshing} className="gap-2">
                  <RefreshCw className="size-4" /> Refresh
                </Button>
              </div>
              <p className="text-sm text-muted-foreground">{models.length} callable models discovered</p>
              <div className="flex flex-wrap gap-2">
                {models.map((model) => (
                  <Badge key={model} variant="outline">
                    {model}
                  </Badge>
                ))}
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Provider quota</CardTitle>
          <CardDescription>
            {usage
              ? `${usage.plan} plan · updated ${new Date(usage.checked_at).toLocaleString()}`
              : "Sign in and select a project to read quota."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {usage?.quotas.map((quota) => (
            <div key={quota.id} className="rounded-md border p-3">
              <div className="flex items-center justify-between gap-4 text-sm">
                <span className="font-medium">{quota.display_name ?? quota.id}</span>
                <span>{formatQuota(quota)}</span>
              </div>
              {quota.reset_at && (
                <p className="mt-1 text-xs text-muted-foreground">Resets {new Date(quota.reset_at).toLocaleString()}</p>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
