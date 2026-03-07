import { useLocation } from "react-router";

export function MainPage() {
  const location = useLocation();
  const version = (location.state as { version?: string })?.version;

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="text-center">
        <h1 className="text-2xl font-bold text-gray-900">
          PaperChat is running
        </h1>
        {version && (
          <p className="mt-2 text-sm text-gray-500">v{version}</p>
        )}
      </div>
    </div>
  );
}
