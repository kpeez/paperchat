import type { CheckResult } from "../api/bootstrap";

interface Props {
  name: string;
  check: CheckResult;
}

export function BootstrapStatus({ name, check }: Props) {
  return (
    <div className="flex items-center gap-2 py-1">
      <span
        className={`inline-block h-2.5 w-2.5 rounded-full ${
          check.ok ? "bg-green-500" : "bg-red-500"
        }`}
      />
      <span className="text-sm font-medium text-gray-700">{name}</span>
      {check.message && (
        <span className="text-sm text-gray-500">— {check.message}</span>
      )}
    </div>
  );
}
