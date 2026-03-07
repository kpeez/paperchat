import type { BootstrapError as BootstrapErrorType } from "../api/bootstrap";

interface Props {
  error: BootstrapErrorType;
}

export function BootstrapError({ error }: Props) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4">
      <p className="text-sm font-semibold text-red-800">{error.code}</p>
      <p className="mt-1 text-sm text-red-700">{error.message}</p>
      <p className="mt-2 text-xs text-red-600">{error.action}</p>
    </div>
  );
}
