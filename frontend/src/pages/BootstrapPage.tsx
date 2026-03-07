import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import {
  fetchHealth,
  fetchBootstrap,
  type BootstrapResponse,
} from "../api/bootstrap";
import { BootstrapStatus } from "../components/BootstrapStatus";
import { BootstrapError } from "../components/BootstrapError";

type Phase = "connecting" | "checking" | "error";

export function BootstrapPage() {
  const navigate = useNavigate();
  const [phase, setPhase] = useState<Phase>("connecting");
  const [bootstrap, setBootstrap] = useState<BootstrapResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    let connected = false;

    async function check() {
      try {
        if (!connected) {
          await fetchHealth();
          if (cancelled) return;
          connected = true;
          setPhase("checking");
        }

        const data = await fetchBootstrap();
        if (cancelled) return;

        if (data.status === "ready") {
          navigate("/app", { state: { version: data.app_version } });
        } else {
          setBootstrap(data);
          if (data.errors.length > 0) {
            setPhase("error");
          }
        }
      } catch {
        // Backend not ready yet, keep polling
      }
    }

    const poll = setInterval(check, 1000);

    return () => {
      cancelled = true;
      clearInterval(poll);
    };
  }, [navigate]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="w-full max-w-md space-y-6 p-8">
        <h1 className="text-center text-2xl font-bold text-gray-900">
          PaperChat
        </h1>

        {phase === "connecting" && (
          <p className="text-center text-sm text-gray-500">
            Connecting to backend...
          </p>
        )}

        {phase === "checking" && !bootstrap && (
          <p className="text-center text-sm text-gray-500">
            Checking services...
          </p>
        )}

        {bootstrap && (
          <>
            <div className="space-y-1">
              {Object.entries(bootstrap.checks).map(([name, check]) => (
                <BootstrapStatus key={name} name={name} check={check} />
              ))}
            </div>

            {bootstrap.errors.length > 0 && (
              <div className="space-y-3">
                {bootstrap.errors.map((err) => (
                  <BootstrapError key={err.code} error={err} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
